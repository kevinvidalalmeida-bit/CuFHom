"""Build scientific CSV records and plots from measured JSON/CUPTI evidence."""
import argparse,csv,json,math,sqlite3,re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_fiber_audit import build_fibers

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def clean_svg(directory):
    # Matplotlib path formatting adds trailing spaces; newlines already
    # separate path coordinates, so removing those spaces preserves the figure.
    for path in directory.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

def stats(v):
    v=np.asarray(v,float)
    return dict(median_ms=float(np.median(v)),mean_ms=float(v.mean()),std_ms=float(v.std(ddof=1)) if v.size>1 else '',
                minimum_ms=float(v.min()),p95_ms=float(np.percentile(v,95)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    load=lambda path:json.loads(path.read_text())
    names=['N32','N48','N64','N96','N128','N192','N256','G08','G09']
    names+=sorted(p.name for p in a.input.iterdir() if p.is_dir() and
        re.fullmatch(r'(?:laminate|regions|random_inclusions|compact_c[\d.]+|sfrp_vf\d+_ar[\d.]+)_N\d+',p.name) and (p/'report.json').exists())
    data={n:load(a.input/n/'report.json') for n in names}
    trace=load(a.input/'trace_summary.json');cal=load(a.input/'calibration_with_cache.json')
    bw=min(cal['bandwidth_copy']['GB_s'],cal['bandwidth_triad']['GB_s'])*1e9
    l2bw=cal['bandwidth_cache_resident']['GB_s']*1e9;l2=cal['device_properties']['l2CacheSize']
    fp32=cal['arithmetic']['float']['TFLOP_s']*1e12;fp64=cal['arithmetic']['double']['TFLOP_s']*1e12
    latency=cal['empty_graph']['per_kernel_s']
    baseline=[];scaling=[];stages=[];kernels=[];breakdowns=[];amdahl=[];ideal=[];opportunities=[];models={}
    iteration_statistics=[]
    for name,r in data.items():
        time=stats([v['wall_s']*1000 for v in r['runs']]);it=int(np.median([v['iterations'] for v in r['runs']]))
        family=(r.get('analytic_family') or 'compact') if name not in ('G08','G09') else name
        contrast=r.get('contrast_longitudinal',r['material']['Ef_L']/r['material']['Em'])
        suite=f'{family}_c{contrast:g}' if name not in ('G08','G09') else name
        if 'fiber_metadata' in r:suite=r['fiber_metadata']['case']
        row=dict(case=name,suite=suite,N=r['shape'][0],voxels=r['voxels'],vf=r['vf'],repeats=len(r['runs']),**time,
            iterations=it,coordinate_iterations=r['runs'][0]['coordinate_iterations'],correction_iterations=r['runs'][0]['correction_iterations'],
            inner_cg_diagnostic_ms=r['diagnostic']['inner_cg_s']*1000,
            inner_cg_per_coordinate_iteration_ms=r['diagnostic']['inner_cg_s']*1000/r['diagnostic']['coordinate_iterations'],
            import_ms=r['import_s']*1000,context_ms=r['context_s']*1000,
            geometry_read_or_generation_ms=r['input_generation_or_read_s']*1000,geometry_compile_ms=r['geometry_compile_s']*1000,
            first_solve_existing_caches_ms=r['first_call']['wall_s']*1000,
            memory_sampled_active_MiB=r['peak_active_pool_bytes']/2**20,reserved_pool_MiB=r['reserved_pool_bytes']/2**20,
            min_driver_free_GiB=r['minimum_driver_free_bytes']/2**30,
            max_physical_residual=max(v['max_residual'] for v in r['runs']),
            max_Ceff_relative_difference=max(v['Ceff_relative_difference'] for v in r['runs']),
            min_Ceff_eigenvalue=min(v['minimum_Ceff_eigenvalue'] for v in r['runs']),
            scope='compiled geometry; six loads; FP32 fields; FP64 reductions; physical rtol 1e-5; no exports')
        repeated_path=a.input/f'iteration_{name}'/'report.json'
        row['inner_cg_timing_repeats']=1
        if repeated_path.exists():
            repeated=load(repeated_path);row['inner_cg_timing_repeats']=len(repeated['runs'])
            row['inner_cg_diagnostic_ms']=repeated['statistics_s']['coordinate_cg_s']['median']*1000
            row['inner_cg_per_coordinate_iteration_ms']=repeated['statistics_s']['coordinate_cg_per_iteration_s']['median']*1000
            row['fallback_cg_diagnostic_ms']=repeated['statistics_s']['correction_cg_s']['median']*1000
            for quantity in ('coordinate_cg_s','correction_cg_s','coordinate_cg_per_iteration_s'):
                iteration_statistics.append(dict(case=name,quantity=quantity,repeats=len(repeated['runs']),
                    **stats([run[quantity]*1000 for run in repeated['runs']]),
                    scope='separate synchronized diagnostic; no setup/postprocess; not additive to uninstrumented baseline'))
        baseline.append(row)
        if 'fiber_metadata' in r:
            fm=r['fiber_metadata']
            row.update(Vf_target=fm['Vf_target'],AR=fm['AR'],box_um=fm['box_um'],length_um=fm['length_um'],
                voxels_per_um=fm['voxels_per_um'],voxels_per_diameter=fm['voxels_per_diameter'],native_resolution_6=fm['native_resolution_6'],
                binary_diameter_below_2_voxels=fm['binary_diameter_below_2_voxels'])
        sc=dict(case=name,suite=suite,N=r['shape'][0],voxels=r['voxels'],stored_real_dof=r['stored_real_dof'],
            admissible_real_dof=r['admissible_real_dof'],coordinate_real_slots=r['coordinate_real_slots'],
            median_total_ms=time['median_ms'],coordinate_iterations=row['coordinate_iterations'],iterations=it,
            diagnostic_inner_cg_ms=row['inner_cg_diagnostic_ms'],diagnostic_per_iteration_ms=row['inner_cg_per_coordinate_iteration_ms'],
            sampled_active_MiB=row['memory_sampled_active_MiB'],reserved_pool_MiB=row['reserved_pool_MiB'],
            included_in_scaling_fit=name not in ('G08','G09') and
                ('fiber_metadata' not in r or r['shape'][0] in (32,48,64,96,128,192,256)))
        scaling.append(sc)
        for component in r['runs'][0]['stages_s']:
            v=[x['stages_s'][component]*1000 for x in r['runs']]
            st=dict(case=name,component=component,clock_domain='synchronized host stages; mean is additive',**stats(v),
                fraction_mean=float(np.mean(v))/time['mean_ms'],calls=6 if component=='six_load_solves' else 1,
                mean_ms_per_call=float(np.mean(v))/(6 if component=='six_load_solves' else 1))
            stages.append(st);sc[component+'_ms']=st['median_ms']
    # Exclusive eight-part timeline: GPU activity first, then API occupied idle, then host gaps.
    components={
        'T_setup':['idle_host_setup_material','setup_material_kernels'],
        'T_memory':['memory_memset','memory_device_copy','idle_in_alloc_api'],
        'T_operator':['operator_fft','operator_other'],
        'T_solver':['solver_update','solver_elementwise','idle_host_in_cg','idle_in_launch_api','idle_in_sync_api'],
        'T_reductions':['reductions'],
        'T_transfers':['transfers','idle_in_memcpy_api'],
        'T_postprocess':['postprocess_kernels','idle_host_postprocess'],
        'T_other':['outside_cg_kernels','idle_host_outside_cg','idle_in_other_api']}
    for name,t in trace.items():
        wall=t['wall_ms'];N=data[name]['shape'][0];V=N**3;H=N*N*(N//2+1)
        assert abs(sum(t['exclusive_ms'].values())-wall)<1e-6
        for component,groups in components.items():
            cost=sum(t['exclusive_ms'].get(g,0.) for g in groups)
            variants=[v for k in t['kernels'] for v in k['variants'] if v['group'] in groups]
            calls=sum(v['calls'] for v in variants)
            cuda=sum(v['calls']*v['mean_us']/1000 for v in variants)
            breakdowns.append(dict(case=name,component=component,exclusive_ms=cost,fraction_wall=cost/wall,
                scope='separate instrumented NVTX TOTAL; GPU/API overlaps removed',calls=calls,
                time_per_call_ms=cuda/calls if calls else '',gpu_kernel_ms=cuda,
                call_definition='GPU kernel calls; per-call duration excludes host/API parts of the component',
                definition=' + '.join(groups)))
        assert abs(sum(r['exclusive_ms'] for r in breakdowns if r['case']==name)-wall)<1e-6
        cumulative=0.;kernel_sum=sum(k['total_ms'] for k in t['kernels']);floor_fixed=0.;floor_model=0.;coverage=0.
        for k in t['kernels']:
            namek=k['kernel'];count=k['calls'];b=f=d=None;basis='unavailable';limiter='unresolved'
            if 'fft' in namek:
                batch=1 if N==240 else 6
                b=(4*V+8*H)*batch if ('r2c' in namek or 'c2r' in namek) else 16*H*batch
                f=(2.5*V if ('r2c' in namek or 'c2r' in namek) else 5*H)*math.log2(N)*batch;d=0.
                basis='1D reference FFT operation/I-O model; actual cuFFT traffic/instructions unmeasured'
                limiter='large: bandwidth/FFT layout candidate; small: device latency candidate; see sampled counters when present'
            elif namek=='indexed_sym21_apply':
                b=52*V;f=66*V;d=0.;basis='SoA six-input/six-output + int32 ID; table assumed cached; 66 scalar ops/voxel'
                limiter='bandwidth candidate from low logical arithmetic intensity'
            elif namek=='coordinates':
                # Register metadata distinguishes direction fusion in this source build.
                fraction=sum(v['calls'] for v in k['variants'] if v['registers']==40)/count
                b=(72+48*fraction)*H;f=(150+12*fraction)*H;d=0.
                basis='coordinate-map I/O estimate; direction variant classified by registers=40; arithmetic count approximate'
                limiter='bandwidth + divide/sqrt/integer indexing candidate'
            elif namek=='adjoint_dot':
                b=96*H+8*math.ceil(H/256);f=150*H;d=12*H
                basis='six complex inputs, three outputs, direction dot; FP64 partial; map arithmetic approximate'
                limiter='bandwidth/FP64 reduction candidate'
            elif namek=='update_rr':
                fraction=sum(v['calls'] for v in k['variants'] if v['registers']==36)/count
                b=(144+48*fraction)*H;f=(24+18*fraction)*H;d=12*H
                basis='six FP32-coordinate vector read/write equivalents; +two smoothing equivalents for registers=36'
                limiter='bandwidth/FP64 reduction candidate'
            elif namek=='warp_dot':
                # A mixture of physical fields and coordinates. Use per-variant NVTX tags.
                post=sum(v['calls'] for v in k['variants'] if v['group']=='postprocess_kernels')/count
                b=48*(post*V+(1-post)*H);f=0.;d=12*(post*V+(1-post)*H)
                basis='FP64 products/sums; outside-postprocess mixes physical and coordinate norm lengths: estimate'
                limiter='bandwidth/FP64 reduction candidate'
            real=k['mean_us']/1e6;fixed=sum(v['calls']*v['p10_us'] for v in k['variants'])/1e6
            floor_fixed+=fixed
            model=None;nocache=None;efficiency=None;potential=None
            if b is not None:
                nocache=max(b/bw,f/fp32,d/fp64,latency)
                # Sensitivity model, NOT a measured DRAM lower bound: optimistic
                # initial/final L2 residency capped by 2*L2 per launch.
                model=max(max(0.,b-2*l2)/bw,b/l2bw,f/fp32,d/fp64,latency)
                efficiency=model/real;potential=real/model;floor_model+=model*count
                coverage+=k['total_ms']
            else:floor_model+=latency*count
            cumulative+=k['total_ms']
            kr=dict(case=name,kernel=namek,calls=count,total_cuda_ms=k['total_ms'],mean_cuda_us=k['mean_us'],
                fraction_profile_wall=k['fraction_wall'],fraction_cuda=k['total_ms']/kernel_sum,cumulative_cuda_fraction=cumulative/kernel_sum,
                logical_bytes_per_call=b,estimated_FP32_FLOPs_per_call=f,estimated_FP64_FLOPs_per_call=d,
                estimated_arithmetic_intensity=(f+d)/b if b else '',
                logical_GB_s=b/real/1e9 if b else '',FP32_TFLOP_s=f/real/1e12 if f else '',FP64_TFLOP_s=d/real/1e12 if d else '',
                roof_no_cache_model_us=nocache*1e6 if nocache is not None else '',
                roof_optimistic_cache_model_us=model*1e6 if model is not None else '',
                efficiency_cache_model=efficiency if efficiency is not None else '',potential_speedup_cache_model=potential if potential is not None else '',
                fixed_kernel_p10_sum_ms=fixed*1000,basis=basis,classification=limiter,
                registers_per_thread_min=k['registers_per_thread_min'],registers_per_thread_max=k['registers_per_thread_max'],
                local_memory_per_thread_max=k['local_memory_per_thread_max'],shared_memory_max=k['shared_memory_bytes_max'],
                achieved_occupancy='',DRAM_bytes='',DRAM_throughput='',L2_hit_rate='',warp_efficiency='',dominant_stalls='',
                counters_status='ERR_NVGPUCTRPERM',roof_status=('calibration roof exceeded: invalid bound' if efficiency is not None and efficiency>1
                    else 'conditional static/cache model; not a certified speedup ceiling'))
            kernels.append(kr)
        models[name]=dict(profile_wall_ms=wall,baseline_ms=next(r['median_ms'] for r in baseline if r['case']==name),
            cuda_kernel_ms=kernel_sum,kernel_busy_fraction=kernel_sum/wall,modeled_cuda_coverage=coverage/kernel_sum,
            optimistic_cache_kernel_model_ms=floor_model*1000,fixed_kernel_p10_floor_ms=floor_fixed*1000,
            fixed_kernel_ratio=wall/(floor_fixed*1000),cache_model_ratio=wall/(floor_model*1000),
            invalid_cache_rows=[k['kernel'] for k in kernels if k['case']==name and k['efficiency_cache_model']!='' and k['efficiency_cache_model']>1],
            qualifier='Floors exclude setup/transfers/host control and are optimistic scenarios. '
                'P10 is a measured-duration extrapolation keeping the current kernels; cache roof is conditional. Neither certifies global hardware minimum.')
        # Amdahl uses non-overlapping GPU durations/profile wall (same clock domain).
        groups={'cuFFT':t['exclusive_ms']['operator_fft'],'operator_total':sum(t['exclusive_ms'].get(x,0.) for x in ('operator_fft','operator_other')),
            'solver_update':t['exclusive_ms']['solver_update'],'explicit_reductions':t['exclusive_ms']['reductions'],
            'postprocess':sum(t['exclusive_ms'].get(x,0.) for x in ('postprocess_kernels','idle_host_postprocess')),
            'largest_kernel':t['kernels'][0]['total_ms']}
        for group,cost in groups.items():
            fraction=cost/wall
            for speed in (2,5,10,float('inf')):
                total=wall-cost+(0. if math.isinf(speed) else cost/speed)
                amdahl.append(dict(case=name,component=group,component_ms=cost,fraction_wall=fraction,
                    component_speedup='infinite' if math.isinf(speed) else speed,total_ms=total,global_speedup=wall/total,
                    scope='Amdahl hypothetical; profiled wall, no overlap; no accuracy claim'))
        api=t['api'];exclusive=t['exclusive_ms']
        host_idle=sum(v for k,v in exclusive.items() if k.startswith('idle_'))
        actual_trans=exclusive.get('transfers',0.)
        launch_idle=exclusive.get('idle_in_launch_api',0.)
        alloc_idle=exclusive.get('idle_in_alloc_api',0.)
        for scenario,cost in [('current',0.),('no_actual_transfers',actual_trans),('no_cuda_allocations',alloc_idle),
            ('no_launch_api_idle',launch_idle),('zero_largest_kernel',t['kernels'][0]['total_ms']),('no_host_idle',host_idle)]:
            ideal.append(dict(case=name,scenario=scenario,total_ms=wall-cost,removed_ms=cost,
                scope='optimistic subtraction from exclusive trace; dependent decisions may prevent attainment'))
        for util in (.7,.8,.9):
            modeled=[k for k in kernels if k['case']==name and k['logical_bytes_per_call'] is not None]
            # Replacing matched modeled kernels only; unknown costs retain their measured time.
            total=wall-sum(k['total_cuda_ms'] for k in modeled)
            for k in modeled:
                b=k['logical_bytes_per_call'];f=k['estimated_FP32_FLOPs_per_call'];d=k['estimated_FP64_FLOPs_per_call']
                tm=max(max(0,b-2*l2)/(util*bw),b/l2bw,f/fp32,d/fp64,latency)
                total+=tm*k['calls']*1000
            ideal.append(dict(case=name,scenario=f'cache_model_{int(util*100)}pct_sustainable_BW',total_ms=total,removed_ms=wall-total,
                scope='conditional optimistic L2-residency/FFT-I-O model; does not use sampled DRAM counters'))
        def opportunity(title,cost,best,evidence,difficulty,experiment):
            opportunities.append(dict(case=name,opportunity=title,current_cost_ms=cost,fraction_total=cost/wall,
                estimated_best_cost_ms=best,max_global_speedup=wall/(wall-cost+best),evidence=evidence,
                difficulty=difficulty,recommended_experiment=experiment))
        fftbest=sum(k['fixed_kernel_p10_sum_ms'] for k in kernels if k['case']==name and 'fft' in k['kernel'])
        opportunity('cuFFT current kernels at observed P10; remove host gaps separately',groups['cuFFT'],fftbest,
            'CUPTI kernel durations; observed P10 grouped by kernel registers/phase; extrapolation, not a new solver result',
            'high','Compare FFT policies using full-solve statistics and sampled DRAM/L2 evidence; retain exact transforms and discretization')
        opportunity('GPU-resident CG scalar decisions / remove CPU gaps',host_idle,0.,
            'Exclusive CUPTI/NVTX host idle; absolute optimistic ceiling, includes mandatory CPU setup',
            'high','Small-grid diagnostic with device FP64 reductions and unchanged true physical audit')
        opportunity('Eliminate actual CPU/GPU data movement',actual_trans,0.,
            'CUPTI transfer device time; inclusive cudaMemcpy API wait is excluded',
            'medium','Keep material IDs/data on GPU across repeated materials; preserve host output')
        opportunity('Remove CUDA allocations inside warm solve',alloc_idle,0.,
            'No malloc/free in warm CUPTI trace; Python pool requests remain; no predicted GPU allocation benefit',
            'low','Persistent-buffer microbenchmark already run; measure host pool requests separately')
        opportunity('Zero-cost postprocess upper ceiling',groups['postprocess'],0.,
            'NVTX/CUPTI exclusive postprocess; hypothetical zero cost ceiling',
            'medium','Low priority in large grids by measured Amdahl fraction')
    fitrows=[]
    for suite in sorted(set(r['suite'] for r in scaling if r['included_in_scaling_fit'])):
        for subset in ('all','N>=96','N>=128'):
            selected=[r for r in scaling if r['included_in_scaling_fit'] and r['suite']==suite and
                (subset=='all' or r['N']>=int(subset[3:]))]
            if len(selected)<3:continue
            for key in ('median_total_ms','diagnostic_inner_cg_ms','diagnostic_per_iteration_ms','six_load_solves_ms','postprocess_ms','material_setup_ms'):
                x=np.log([r['N'] for r in selected]);y=np.log([r[key] for r in selected]);coef=np.polyfit(x,y,1);prediction=np.polyval(coef,x)
                r2=1-float(np.sum((y-prediction)**2)/np.sum((y-y.mean())**2))
                fitrows.append(dict(suite=suite,subset=subset,quantity=key,p=float(coef[0]),a_ms=float(np.exp(coef[1])),R2_log=r2,points=len(selected)))
    for name in ('G08','G09'):
        r=data[name];frac=r['diagnostic']['inner_cg_s']/r['diagnostic']['wall_s']
        for reduction in (.25,.5,.75):
            ideal.append(dict(case=name,scenario=f'CG_iterations_reduced_{int(reduction*100)}pct',
                total_ms=next(x['median_ms'] for x in baseline if x['case']==name)*(1-frac*reduction),
                removed_ms='',scope='Hypothetical same physical tolerance; no preconditioner setup/apply cost. Not an observed speedup.'))
    # Paired, alternating full-solve experiments are separate from Amdahl ceilings.
    for path in sorted(a.input.glob('*/report.json')):
        r=load(path)
        if 'median_s' not in r or not r.get('runs'):continue
        reference='auto' if 'auto' in r['median_s'] else 'current'
        if reference not in r['median_s']:continue
        current=r['median_s'][reference]*1000
        for variant,best in r['median_s'].items():
            if variant==reference:continue
            opportunities.append(dict(case=path.parent.name,opportunity=f'Isolated full solve: {variant}',
                current_cost_ms=current,fraction_total=1.,estimated_best_cost_ms=best*1000,
                max_global_speedup=current/(best*1000),
                evidence=f'{path.parent.name}/report.json; alternating paired medians; same physical residual; observed ratio, not maximum',
                difficulty='medium',decision=('rejected: matched-precision dual control slower' if variant.startswith('dual') else
                    'candidate by grid only; review full fields' if best<r['median_s'][reference]*.99 else 'no measured improvement'),
                numerical_comparability=('public tolerance preserved; public dual accepts a larger actual residual; half-rtol control slower'
                    if variant.startswith('dual') else 'same iterations and physical criterion; full field validations in field_validation.csv'),
                recommended_experiment=('Do not integrate; compare half-rtol control and full fields' if variant.startswith('dual') else
                    'Review six complete field errors and repeat across contrasting geometries before changing defaults')))
    opportunities.sort(key=lambda r:(r.get('decision','').startswith('rejected'),
        -(r['current_cost_ms']-r['estimated_best_cost_ms'])))
    counter_path=a.output/'gpu_counters.csv'
    if counter_path.exists():
        from build_gpu_counters import attach_counters
        with counter_path.open() as f:attach_counters(kernels,list(csv.DictReader(f)))
    for file,rows in [('performance_baseline.csv',baseline),('scaling_results.csv',scaling),('kernel_profile.csv',kernels),
        ('optimization_opportunities.csv',opportunities),('component_breakdown.csv',breakdowns),('stage_breakdown.csv',stages),
        ('amdahl.csv',amdahl),('ideal_solver_scenarios.csv',ideal),('scaling_fits.csv',fitrows),
        ('iteration_statistics.csv',iteration_statistics)]:
        if rows:write_csv(a.output/file,rows)
    (a.output/'models.json').write_text(json.dumps(dict(models=models,calibration=cal,scaling_fits=fitrows),indent=2)+'\n')
    write_csv(a.output/'cuda_api_profile.csv',[dict(case=name,api=api,**record,
        scope='inclusive API duration overlaps CUDA kernels; do not add to wall breakdown')
        for name,t in trace.items() for api,record in t['api'].items()])
    write_csv(a.output/'cuda_memory_profile.csv',[dict(case=name,operation=operation,**record,
        scope='actual device copy/memset duration and bytes; API wait is separate')
        for name,t in trace.items() for operation,record in t['transfers'].items()])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    build_fibers(a.input,a.output,baseline,scaling,write_csv)
    synthetic=[r for r in scaling if r['included_in_scaling_fit'] and not r['suite'].startswith('sfrp_')]
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for suite in sorted(set(r['suite'] for r in synthetic)):
        rs=sorted((r for r in synthetic if r['suite']==suite),key=lambda r:r['N']);n=[r['N'] for r in rs]
        ax[0].loglog(n,[r['median_total_ms'] for r in rs],'o-',label=suite)
        ax[1].loglog(n,[r['diagnostic_per_iteration_ms'] for r in rs],'o-',label=suite)
        ax[2].plot(n,[r['iterations'] for r in rs],'o-',label=suite)
    ax[0].set(xlabel='Grid side N',ylabel='Six-load wall time [ms]',title='General geometries');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Grid side N',ylabel='Inner CG / iteration [ms]',title='Iteration cost')
    ax[2].set(xlabel='Grid side N',ylabel='Iterations over six loads',title='Iteration count is separate from cost')
    fig.tight_layout();fig.savefig(a.output/'scaling.png',dpi=180);fig.savefig(a.output/'scaling.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));namestrace=['G08','N64','G09','N256'];bottom=np.zeros(4)
    labels={'T_setup':'Setup','T_memory':'Memory','T_operator':'Operator','T_solver':'CG updates + host control',
        'T_reductions':'Explicit reductions','T_transfers':'Transfer + idle in memcpy API','T_postprocess':'Postprocess','T_other':'Other'}
    for component in components:
        vals=np.array([next(r['fraction_wall'] for r in breakdowns if r['case']==n and r['component']==component)*100 for n in namestrace])
        ax.bar(namestrace,vals,bottom=bottom,label=labels[component]);bottom+=vals
    ax.set(ylabel='Share of profiled wall time [%]',title='Exclusive timeline: each instant is counted once',ylim=(0,100))
    ax.legend(loc='center left',bbox_to_anchor=(1, .5),fontsize=9);fig.tight_layout()
    fig.savefig(a.output/'breakdown.png',dpi=180);fig.savefig(a.output/'breakdown.svg');plt.close(fig)
    clean_svg(a.output)
    print(json.dumps(dict(models=models,scaling_fits=fitrows),indent=2))
if __name__=='__main__':main()
