"""Rebuild the Vf/AR campaign and code-review measurements offline."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASELINE='9300f9302064e3b4feb88a63eb8b84cd36beb65b'
GRIDS=[32,48,64,96,128,192,256]
CASES=['G08','G09','F192','N64','N256','R128']


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def collect(curves,pilots):
    files={}
    for path in sorted(curves.rglob('*.json')):
        files['curve/'+str(path.relative_to(curves))]=json.loads(path.read_text())
    for pattern in ('verified_*','production_*'):
        for directory in sorted(pilots.glob(pattern)):
            for path in sorted(directory.glob('*.json')):
                files['pilot/'+str(path.relative_to(pilots))]=json.loads(path.read_text())
    sources={};history={}
    paths=list((ROOT/'src').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
    paths += [ROOT/'benchmarks/profile_g09_linear.py',ROOT/'benchmarks/audit_linear_performance.py',
              ROOT/'benchmarks/linear_strategy_20261002/validate_fields.py']
    paths+=list((ROOT/'tests').glob('test_material_rotation.py'))+list((ROOT/'tests').glob('test_linear_final_audit.py'))
    for path in sorted(set(paths)):
        name=str(path.relative_to(ROOT));value=path.read_text()
        sources[name]=dict(text=value,sha256=digest(value.encode()))
        if path.is_relative_to(ROOT/'src'):
            old=subprocess.run(['git','show',f'{BASELINE}:{name}'],cwd=ROOT,text=True,capture_output=True)
            if old.returncode==0 and old.stdout!=value:
                history[digest(old.stdout.encode())]=dict(path=name,text=old.stdout)
    masters={}
    for name,case in files['curve/campaign.json']['cases'].items():
        data=(ROOT/case['table']).read_bytes();assert digest(data)==case['master_sha256']
        masters[name]=dict(path=case['table'],text=data.decode(),sha256=digest(data))
    logs={}
    for name,path in dict(rotation='/tmp/cufhom_vf_ar_rotation_tests.txt',
        production='/tmp/cufhom_vf_ar_full_tests.txt',specific='/tmp/cufhom_vf_ar_specific_tests.txt').items():
        if Path(path).exists():logs[name]=Path(path).read_text()
    return dict(schema=1,baseline_commit=BASELINE,files=files,source_snapshot=sources,
        historical_sources=history,continuous_masters=masters,test_logs=logs,
        documents={'CODE_REVIEW.md':Path(__file__).with_name('CODE_REVIEW.md').read_text()},
        exclusions='Debug timing_G08, debug_G09 and fields_N8 lacked spectrum activation and are excluded.',
        raw_roots=dict(curves=str(curves.resolve()),pilots=str(pilots.resolve())))


def summarize(b):
    files=b['files'];campaign=files['curve/campaign.json']
    known={s['sha256'] for s in b['source_snapshot'].values()}|set(b['historical_sources'])
    for v in b['source_snapshot'].values():assert digest(v['text'].encode())==v['sha256']
    for sha,v in b['historical_sources'].items():assert digest(v['text'].encode())==sha
    def sources(r):
        for key in ('source_sha256','pilot_source_sha256'):
            for sha in r.get(key,{}).values():assert sha in known,sha
        if 'benchmark_sha256' in r:assert r['benchmark_sha256'] in known
    def physics(r,directory):
        assert r['inputs_unchanged'] and r['load_ids']==list(range(6))
        for row in r['runs']+r['warmups']:
            C=np.asarray(row['Ceff']);assert np.linalg.eigvalsh((C+C.T)/2)[0]>0
            assert row['max_physical_residual']<=1e-5 and row['Ceff_relative_difference']<=1e-6
            detail=files[f"{directory}/{row['variant']}_{row['repetition']}.json"]
            loads=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            assert len(loads)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in loads)
    assert campaign['baseline_commit']==BASELINE and len(campaign['completed'])==72 and campaign['repeats']==5
    assert campaign['runner_sha256'] in known
    assert len(campaign['cases'])==len(b['continuous_masters'])==9
    rows=[];samples=[];plans=[]
    for choice in files['curve/fft_cache.json']['choices'].values():
        c=choice['configuration']
        plans.append(dict(N=c['shape'][0],policy=choice['policy'],tuning_s=choice['tuning_s'],
            gpu=c['gpu'],runtime=c['runtime'],driver=c['driver'],cufft=c['cufft']))
    for name in sorted(campaign['completed']):
        directory='curve/'+name;r=files[directory+'/report.json'];sources(r);physics(r,directory)
        assert r['source_sha256']==campaign['source_sha256'] and len(r['runs'])==5 and len(r['warmups'])==1
        f=r['fiber_metadata'];m=b['continuous_masters'][f['case']]
        assert digest(m['text'].encode())==m['sha256']==f['master_sha256']
        assert r['fiber_case']['periodic_audit']['accepted'] and f['box_um']==2*f['length_um']
        times=[x['wall_s'] for x in r['runs']];details=[]
        for run in r['runs']:
            d=files[f"{directory}/defaults_{run['repetition']}.json"]
            assert d['linear_fft_autotune'] and d['linear_energy_update'] and d['linear_final_audit_reuse']
            load=d['cufhom_core_application_timing']['primaldual']['primal']
            coord=sum(i['coordinate_iterations'] for i in load['load_solver_infos'])
            cleanup=sum(i['correction_iterations'] for i in load['load_solver_infos']);assert coord+cleanup==run['cg']
            plan=load['load_solver_infos'][0]['linear_fft_tuning']['policy']
            details.append(dict(cfield_s=d['cfield_build_s'],load_solve_s=load['load_solve_s'],
                postprocess_s=load['postprocess_s'],cg=run['cg'],coordinate=coord,cleanup=cleanup,plan=plan))
            samples.append(dict(case=f['case'],N=f['N'],repetition=run['repetition'],wall_s=run['wall_s'],
                CG=run['cg'],coordinate_CG=coord,correction_CG=cleanup,physical_residual=run['max_physical_residual']))
        assert len({d['plan'] for d in details})==1
        rows.append(dict(case=f['case'],Vf_target_percent=f['Vf_target']*100,
            Vf_continuous_percent=r['fiber_case']['continuous_capsule_vf']*100,Vf_voxel_percent=f['Vf_voxel']*100,
            AR=f['AR'],N=f['N'],voxels=f['N']**3,box_um=f['box_um'],diameter_voxels=f['voxels_per_diameter'],
            native_resolution_6=f['native_resolution_6'],diameter_below_2_voxels=f['binary_diameter_below_2_voxels'],
            median_s=float(np.median(times)),min_s=min(times),max_s=max(times),
            relative_range_percent=(max(times)-min(times))/np.median(times)*100,
            CG_median=float(np.median([d['cg'] for d in details])),
            coordinate_CG_median=float(np.median([d['coordinate'] for d in details])),
            correction_CG_median=float(np.median([d['cleanup'] for d in details])),
            load_solve_median_s=float(np.median([d['load_solve_s'] for d in details])),
            material_setup_host_median_s=float(np.median([d['cfield_s'] for d in details])),
            postprocess_median_s=float(np.median([d['postprocess_s'] for d in details])),
            geometry_preparation_s=r['geometry_preparation_s'],fft_policy=details[0]['plan'],
            first_prepared_call_s=r['warmups'][0]['wall_s'],
            max_physical_residual=max(x['max_physical_residual'] for x in r['runs']),
            max_Ceff_repeat_relative=max(x['Ceff_relative_difference'] for x in r['runs']),
            master_sha256=f['master_sha256'],
            orientation_A2_anisotropy=float(np.linalg.norm(np.asarray(f['A2_voxel'])-np.eye(3)/3))))
    assert len(rows)==72 and len(samples)==360
    timings=[];fields=[]
    for case in CASES:
        directory='pilot/verified_timing_'+case;r=files[directory+'/report.json'];sources(r);physics(r,directory)
        reference={x['repetition']:x for x in r['runs'] if x['variant']=='current'};assert len(reference)==5
        assert len(r['activations'])==24
        for a in r['activations']:
            if a['variant'] in ('spectrum','combined'):assert a['spectrum_reuses']==6
            if a['variant'] in ('rotation','combined'):assert a['rotation_groups']>0
        for variant in ('spectrum','rotation','combined'):
            variants=[x for x in r['runs'] if x['variant']==variant]
            assert len(variants)==5 and all(x['cg']==reference[x['repetition']]['cg'] for x in variants)
            gain=[100*(1-x['wall_s']/reference[x['repetition']]['wall_s']) for x in variants]
            setup={v:np.median([files[f'{directory}/{v}_{i}.json']['cfield_build_s'] for i in range(5)]) for v in ('current',variant)}
            timings.append(dict(case=case,variant=variant,current_s=r['median_s']['current'],
                candidate_s=r['median_s'][variant],median_saving_percent=100*(1-r['median_s'][variant]/r['median_s']['current']),
                paired_min_saving_percent=min(gain),paired_max_saving_percent=max(gain),
                positive_pairs=sum(g>0 for g in gain),CG=variants[0]['cg'],
                current_material_host_s=setup['current'],candidate_material_host_s=setup[variant]))
    for case in CASES+['N8']:
        r=files[f'pilot/verified_fields_{case}/report.json'];sources(r)
        for row in r['rows']:
            assert len(row['fields_sha256'])==6 and row['max_physical_residual']<=1e-5
            assert row['Ceff_relative_error']<2e-6 and row['min_Ceff_eigenvalue']>0
            error=max(m['relative_l2'] for m in row['fields']);assert error<2e-4
            if row['variant'] in ('spectrum','combined'):assert row['activations']['spectrum_reuses']==6
            fields.append(dict(case=case,variant=row['variant'],max_field_relative_l2=error,
                Ceff_relative_error=row['Ceff_relative_error'],max_physical_residual=row['max_physical_residual'],
                CG=row['iterations'],fields_sha256=';'.join(row['fields_sha256'])))
    production=[];production_fields=[]
    current_source={p:v['sha256'] for p,v in b['source_snapshot'].items() if p.startswith('src/')}
    for case in CASES:
        directory='pilot/production_timing_'+case;r=files[directory+'/report.json'];sources(r);physics(r,directory)
        assert r['source_sha256']==current_source
        assert digest(r['baseline_rotation_source'].encode())==r['baseline_rotation_sha256']
        old=next(v['text'] for v in b['historical_sources'].values() if v['path']=='src/cufhom/pipeline/fft_solver.py')
        assert r['baseline_rotation_source'] in old and r['baseline_commit']==BASELINE
        base={x['repetition']:x for x in r['runs'] if x['variant']=='before'}
        new=[x for x in r['runs'] if x['variant']=='defaults']
        assert len(base)==len(new)==5 and all(x['cg']==base[x['repetition']]['cg'] for x in new)
        gain=[100*(1-x['wall_s']/base[x['repetition']]['wall_s']) for x in new];m=r['median_s']
        for row in r['runs']+r['warmups']:
            d=files[f"{directory}/{row['variant']}_{row['repetition']}.json"]
            loads=d['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            assert sum(i['linear_final_spectrum_reuse'] for i in loads)==(6 if row['variant']=='defaults' else 0)
        production.append(dict(case=case,before_s=m['before'],defaults_s=m['defaults'],
            saving_percent=100*(1-m['defaults']/m['before']),positive_pairs=sum(x>0 for x in gain),
            paired_min_saving_percent=min(gain),paired_max_saving_percent=max(gain),CG=new[0]['cg']))
    for case in CASES+['N8']:
        r=files[f'pilot/production_fields_{case}/report.json'];sources(r)
        assert r['source_sha256']==current_source
        for row in r['rows']:
            error=max(m['relative_l2'] for m in row['fields'])
            assert len(row['fields_sha256'])==6 and error<2e-4 and row['Ceff_relative_error']<2e-6
            assert row['min_Ceff_eigenvalue']>0 and row['max_physical_residual']<=1e-5
            if row['variant']!='cpu':
                assert sum(i['linear_final_spectrum_reuse'] for i in row['physical_infos'])==(6 if row['variant']=='defaults' else 0)
            production_fields.append(dict(case=case,variant=row['variant'],max_field_relative_l2=error,
                Ceff_relative_error=row['Ceff_relative_error'],physical_residual=row['max_physical_residual'],
                fields_sha256=';'.join(row['fields_sha256'])))
    assert '357 passed' in b['test_logs']['production'] and '7 subtests passed' in b['test_logs']['production']
    assert '28 passed' in b['test_logs']['specific'] and '4 passed' in b['test_logs']['rotation']
    return dict(rows=rows,samples=samples,plans=sorted(plans,key=lambda x:x['N']),timings=timings,fields=fields,
                production=production,production_fields=production_fields)


def csv_file(path,rows):
    with path.open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def figures(s,out):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='cufhom-vf-ar-20261002'
    import matplotlib.pyplot as plt
    colors={5:'#246a8e',15:'#d07a26',25:'#5b8961'}
    def save(fig,name):
        fig.supxlabel('RTX 5070 Ti · FP32/FP64 · residual físico 1e−5 · fuente 9300f93',fontsize=8)
        fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig)
        p=out/(name+'.svg');p.write_text('\n'.join(t.rstrip() for t in p.read_text().splitlines())+'\n')
    for quantity,label,name,log in [('median_s','Tiempo de seis cargas (s)','time_vs_voxels',True),
        ('CG_median','Iteraciones totales, seis cargas','iterations_vs_voxels',False)]:
        fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained',sharey=True)
        for ax,ar in zip(axes,[5,10,20]):
            for vf,color in colors.items():
                rows=sorted([r for r in s['rows'] if r['AR']==ar and r['Vf_target_percent']==vf and r['N'] in GRIDS],key=lambda r:r['N'])
                x=np.array([r['N'] for r in rows]);y=np.array([r[quantity] for r in rows])
                ax.plot(x,y,color=color,lw=1.4,label=f'Vf {vf}%')
                for r in rows:ax.plot(r['N'],r[quantity],'o',color=color,ms=4.5,
                    markerfacecolor='white' if r['diameter_below_2_voxels'] else color)
                if log:ax.fill_between(x,[r['min_s'] for r in rows],[r['max_s'] for r in rows],color=color,alpha=.17)
                else:ax.plot(x,[r['coordinate_CG_median'] for r in rows],color=color,ls=':',alpha=.7)
            ax.set_title(f'AR = {ar}');ax.set_xscale('log',base=2);ax.set_xticks(GRIDS,GRIDS)
            if log:ax.set_yscale('log')
            ax.set_xlabel('N, malla N³');ax.grid(alpha=.2,which='both')
        axes[0].set_ylabel(label);axes[0].legend(frameon=False,fontsize=8)
        detail='Bandas: mínimo–máximo de 5 solves warm' if log else 'Punteadas: CG de coordenadas, sin corrección'
        fig.suptitle(f'CuFHom · dominio 2L · orientación aleatoria\n{detail}; círculos vacíos: diámetro <2 vóxeles',fontsize=11)
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    for ax,n in zip(axes,[96,192]):
        for vf,color in colors.items():
            rows=sorted([r for r in s['rows'] if r['N']==n and r['Vf_target_percent']==vf],key=lambda r:r['AR'])
            ax.errorbar([r['AR'] for r in rows],[r['median_s'] for r in rows],
                yerr=np.array([[r['median_s']-r['min_s'] for r in rows],[r['max_s']-r['median_s'] for r in rows]]),
                marker='o',ms=4,capsize=3,color=color,label=f'Vf {vf}%')
        ax.set_title(f'Misma malla: {n}³');ax.set_xticks([5,10,20]);ax.set_xlabel('AR')
        ax.set_ylabel('Tiempo de seis cargas (s)');ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Vf/AR a igual número de vóxeles\nLa resolución del diámetro cambia con AR',fontsize=11);save(fig,'vf_ar_same_grid')
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    for vf,color in colors.items():
        rows=sorted([r for r in s['rows'] if r['native_resolution_6'] and r['Vf_target_percent']==vf],key=lambda r:r['AR'])
        x=[r['AR'] for r in rows];axes[0].plot(x,[r['median_s'] for r in rows],'o-',color=color,label=f'Vf {vf}%')
        axes[1].plot(x,[r['CG_median'] for r in rows],'o-',color=color)
    for ax in axes:ax.set_xticks([5,10,20],['5\n60³','10\n120³','20\n240³']);ax.set_xlabel('AR y malla N³');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[0].set_ylabel('Tiempo de seis cargas (s)')
    axes[1].set_ylabel('Iteraciones totales, seis cargas');axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Misma resolución: 6 vóxeles/diámetro · dominio 2L\nAl aumentar AR crecen dominio y número de vóxeles',fontsize=11);save(fig,'fixed_resolution')


def render(s):
    lookup={(r['Vf_target_percent'],r['AR'],r['N']):r for r in s['rows']}
    lines=['# CuFHom: curvas de Vf, AR y vóxeles','',
        '**72 puntos, cinco repeticiones por punto, nueve microestructuras continuas.** '
        f'Las curvas usan los defaults de {BASELINE[:12]}, antes de los pilotos de esta revisión. '
        'Las comparaciones de candidatos se miden aparte y sus ahorros no se suman a las curvas.','',
        '## Condiciones','',
        '- Vf objetivo 5, 15 y 25%; AR 5, 10 y 20; orientación aleatoria; d = 1 µm.',
        '- L = AR × d es la longitud axial de la cápsula. Su longitud total incluye las tapas. Dominio cúbico de lado 2L.',
        '- El mismo master se voxeliza en 32³, 48³, 64³, 96³, 128³, 192³ y 256³. A 6 vóxeles por diámetro se usan además 60³, 120³ y 240³.',
        '- RTX 5070 Ti; CuPy 13.6.0, runtime CUDA 12.9, cuFFT 11.4.1. Seis cargas lineales; campos/FFT FP32, productos FP64, residual físico ≤1e-5 y Ceff SPD.',
        '- Un calentamiento por punto y cinco solves medidos: 360 muestras y 72 calentamientos, 2.592 cargas.',
        '- La pared incluye setup del material, solve, auditorías, posprocesado y JSON. Lectura/agrupación de geometría y hashes se hacen antes y se informan aparte.',
        '- Un master por Vf/AR: la dispersión temporal no mide variabilidad entre realizaciones. El escritorio comparte GPU; no se fijan relojes.','',
        '## Curvas','',
        '![Tiempo frente a vóxeles](time_vs_voxels.png)','',
        '![Iteraciones frente a vóxeles](iterations_vs_voxels.png)','',
        'Los círculos vacíos tienen menos de dos vóxeles por diámetro. Son tiempos de una entrada discreta; '
        'no prueban fidelidad geométrica ni convergencia de homogenización. Bandas: mínimo/máximo de cinco repeticiones, '
        'no intervalos de confianza. Las líneas unen mediciones sin ajustar una ley universal.','',
        '### Vf/AR a igual número de vóxeles','',
        '![Vf y AR a igual malla](vf_ar_same_grid.png)','',
        '| Vf % | AR | 96³ s | CG / corrección | 192³ s | CG / corrección | 256³ s | CG / corrección |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for vf in [5,15,25]:
        for ar in [5,10,20]:
            values=[]
            for n in [96,192,256]:
                r=lookup[(vf,ar,n)];values.extend([f"{r['median_s']:.6f}",f"{r['CG_median']:.0f} / {r['correction_CG_median']:.0f}"])
            lines.append('| '+' | '.join([str(vf),str(ar)]+values)+' |')
    lines += ['', '### A igual resolución física','', '![Resolución física constante](fixed_resolution.png)','',
        '| Vf % | AR 5, 60³ s | AR 10, 120³ s | AR 20, 240³ s |','| ---: | ---: | ---: | ---: |']
    for vf in [5,15,25]:
        lines.append('| '+' | '.join([str(vf)]+[f"{lookup[(vf,ar,int(12*ar))]['median_s']:.6f}" for ar in [5,10,20]])+' |')
    lines += ['', 'A igual resolución y dominio 2L, pasar de AR 5 a 20 multiplica el volumen discreto por 64. '
        'El gráfico incluye ese cambio de tamaño; no aísla el efecto de la forma.','',
        '## Qué explican las mediciones','',
        'Vf y AR no ordenan las iteraciones de manera monótona. Vf 5%, AR 10 necesita 293 CG en 128³ y 281 en 192³, '
        'pero tarda más en 192³ porque cada aplicación recorre más datos. Ese caso suma 48 CG de corrección en 256³. '
        'Vf 25%, AR 10 necesita 288 CG en 96³ y 270 en 192³. Una malla pequeña puede necesitar más iteraciones sin consumir más tiempo total.','',
        'Las iteraciones dependen del espectro, de los modos excitados por los RHS, del contraste/anisotropía y del redondeo. '
        'La memoria y el plan FFT afectan sobre todo al coste de cada aplicación. El barrido no mide autovalores ni identifica '
        'una causa única de cada diferencia. La corrección aparece cuando el criterio de coordenadas no basta para el residual físico. '
        'Los cambios de norma/precisión ya ensayados siguen [descartados](../../descartados/performance_audit_20261001/README.md).','',
        'La elección FFT cambia con N:','', '| N | Política aprendida | Tuning inicial s |','| ---: | --- | ---: |']
    for p in s['plans']:lines.append(f"| {p['N']} | {p['policy']} | {p['tuning_s']:.6f} |")
    lines += ['', 'El tuning se hace una vez por configuración y queda fuera de las medianas warm. '
        'Estas cifras no son tiempos de arranque desde un proceso frío. [Muestras](samples.csv), '
        '[72 puntos y metadatos](curves.csv), [planes](fft_plans.csv).','',
        '## Revisión del código: dos pilotos nuevos','',
        '1. Reutilizar Fσ de la comprobación verdadera en la misma proyección física elimina seis R2C por carga reutilizada: '
        '36 transformadas de componentes en las seis cargas. Conserva escala, cero/Nyquist y residual verdadero.',
        '2. Crot = B(R) C B(R)ᵀ en Mandel sustituye cuatro contracciones de un tensor de cuarto orden y 21/36 lanzamientos '
        'de empaquetado por un kernel. La rotación usa FP64 y redondea una vez al dtype solicitado. '
        'Admite anisotropía general y no usa una fórmula exclusiva de fibras.','',
        '| Caso | Referencia s | Ambos pilotos s | Ahorro medianas % | Pares favorables / 5 | CG, ambos |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in s['timings']:
        if r['variant']=='combined':lines.append(f"| {r['case']} | {r['current_s']:.6f} | {r['candidate_s']:.6f} | {r['median_saving_percent']:.2f} | {r['positive_pairs']} | {r['CG']} |")
    lines += ['', 'La ganancia total es pequeña y algunos casos quedan dentro de la dispersión; no disminuyen las iteraciones. '
        'La rotación ahorra aproximadamente 0,8 ms de setup en los casos pequeños. Ese reloj mide la llamada host '
        'con encolado CUDA, no tiempo exclusivo GPU. [Ablación y extremos pareados](candidates.csv).','',
        'Los seis campos completos se comparan en G08, G09, F192, N64, N256 y R128; 8³ también contra CPU64 a residual 1e-10. '
        'Ceff es SPD y los residuales físicos GPU son ≤1e-5. La rotación se compara con el tensor de cuarto orden CPU64 '
        'para una matriz SPD anisótropa, 31 orientaciones, tabla completa/sym21 y FP32/FP64. [Errores y hashes](fields.csv).','',
        '## Implementación final integrada','',
        'Cinco pares alternados con el mismo código final: defaults frente al constructor de material exacto de 9300f93 '
        'y reutilización del espectro desactivada. La rotación anterior se conserva solo en el runner de control.','',
        '| Caso | Antes s | Defaults finales s | Ahorro medianas % | Pares favorables / 5 | CG, ambos |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in s['production']:
        lines.append(f"| {r['case']} | {r['before_s']:.6f} | {r['defaults_s']:.6f} | {r['saving_percent']:.2f} | {r['positive_pairs']} | {r['CG']} |")
    lines += ['', '**357 pruebas y 7 subpruebas de producción pasan.** G09, F192 y N256 mejoran en los cinco pares. '
        'G08 no muestra una mejora total estable; N64/R128 tienen dispersión. Se conservan ecuación, cargas y aceptación física. '
        'Las diferencias de campos por rotación se informan sin afirmar igualdad bit a bit. '
        '[Tiempos finales](production.csv), [campos finales](production_fields.csv).','',
        'La decisión de producción y las prioridades de la revisión están en [CODE_REVIEW.md](CODE_REVIEW.md).','',
        '## Reproducción','',
        '[evidence.json.gz](evidence.json.gz) conserva JSON, fuentes exactas, hashes, nueve masters continuos y pruebas. '
        'Las voxelizaciones grandes siguen en el directorio local de geometrías. El informe se reconstruye sin GPU:','',
        '    .venv/bin/python benchmarks/vf_ar_curves_20261002/build_report.py --evidence benchmarks/vf_ar_curves_20261002/evidence.json.gz --output /tmp/cufhom_curves_rebuilt','',
        'Nueva campaña GPU secuencial:','',
        '    bash benchmarks/vf_ar_curves_20261002/run_gpu.sh benchmarks/vf_ar_curves_20261002/run_curves.py --output benchmarks/results/vf_ar_new','',
        'Los masters conservan la auditoría periódica y la tolerancia histórica de contacto de 1/120. '
        'Cada punto informa Vf voxelizada y A₂; no se sustituyen por los objetivos.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curves',type=Path);p.add_argument('--pilots',type=Path);p.add_argument('--evidence',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    b=json.loads(gzip.decompress(args.evidence.read_bytes())) if args.evidence else collect(args.curves,args.pilots)
    s=summarize(b);data=gzip.compress(encoded(b),mtime=0);(args.output/'evidence.json.gz').write_bytes(data)
    for name,key in [('curves','rows'),('samples','samples'),('fft_plans','plans'),('candidates','timings'),('fields','fields'),
                     ('production','production'),('production_fields','production_fields')]:
        csv_file(args.output/(name+'.csv'),s[key])
    for name,value in b['documents'].items():(args.output/name).write_text(value)
    figures(s,args.output);(args.output/'RESULTS.md').write_text(render(s))
    manifest=dict(schema=1,baseline_commit=BASELINE,points=72,timed_samples=360,loads_including_warmups=2592,
        evidence_sha256=digest(data),source_sha256={p:v['sha256'] for p,v in b['source_snapshot'].items()},
        outputs={p.name:digest(p.read_bytes()) for p in sorted(args.output.iterdir())
                 if p.suffix in ('.csv','.png','.svg','.md')})
    (args.output/'manifest.json').write_bytes(encoded(manifest))
    print(json.dumps(dict(points=72,timed_samples=360,evidence_bytes=len(data))))


if __name__=='__main__':main()
