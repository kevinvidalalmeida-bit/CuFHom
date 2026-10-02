"""Scientific tables and readable short-fiber audit figures."""
import json
import numpy as np
import matplotlib.pyplot as plt


def build_fibers(raw, output, baseline, scaling, write_csv):
    manifest=json.loads((raw/'fiber_geometries/manifest.json').read_text())
    records=[r for r in baseline if r['suite'].startswith('sfrp_')]
    write_csv(output/'fiber_baseline.csv',records)
    geometry=[]
    for c in manifest['cases']:
        A=np.asarray(c['A2_continuous'])
        geometry.append(dict(case=c['case'],Vf_target=c['Vf_target'],AR=c['AR'],
            fiber_count=c['fiber_count'],continuous_capsule_vf=c['continuous_capsule_vf'],
            diameter_um=c['diameter_um'],axis_length_um=c['length_um'],box_um=c['box_um'],
            native_N=round(6*c['box_um']),A2_isotropy_error=float(np.linalg.norm(A-np.eye(3)/3)),
            A11=A[0,0],A22=A[1,1],A33=A[2,2],
            periodic_audit_accepted=c['periodic_audit']['accepted'],
            periodic_max_relative_overlap=c['periodic_audit']['maximum_relative_overlap'],
            fibers_removed_local_copy=len(c.get('periodic_repair',{}).get('removed_original_rows',[])),
            master_sha256=c['master_sha256'],source_type=c['source_type']))
    write_csv(output/'geometry_manifest.csv',geometry)
    # 12.5 is retained as independent evidence, outside the agreed 3x3 matrix.
    agreed=[r for r in records if r['AR'] in (5,10,20)]
    colors={.05:'#2468ad',.15:'#15936f',.25:'#c35226'}
    styles={5:'-',10:'--',20:':'}
    fig,axs=plt.subplots(1,3,figsize=(14,4.6))
    for suite in sorted(set(r['suite'] for r in agreed)):
        rs=sorted((r for r in agreed if r['suite']==suite and
            r['N'] in (32,48,64,96,128,192,256)),key=lambda r:r['N'])
        n=[r['N'] for r in rs];vf=rs[0]['Vf_target'];ar=rs[0]['AR']
        opts=dict(color=colors[vf],linestyle=styles[ar],marker='o',markersize=3)
        label=f'Vf {vf:.0%}, AR {ar:g}'
        axs[0].loglog(n,[r['median_ms']/1000 for r in rs],label=label,**opts)
        axs[1].loglog(n,[r['inner_cg_per_coordinate_iteration_ms'] for r in rs],**opts)
        axs[2].plot(n,[r['iterations'] for r in rs],**opts)
    axs[0].set(xlabel='Grid side N',ylabel='Six-load wall time [s]',title='Warm median, 9 repetitions')
    axs[1].set(xlabel='Grid side N',ylabel='Coordinate CG / iteration [ms]',title='Iteration cost')
    axs[2].set(xlabel='Grid side N',ylabel='Iterations over six loads',title='Includes physical correction CG')
    axs[0].legend(fontsize=8,ncol=2)
    for ax in axs:ax.grid(alpha=.18)
    for ax in axs[:2]:
        ax.set_xticks([32,64,128,256],labels=['32','64','128','256'])
        ax.set_xticks([],minor=True)
    fig.suptitle('Short fibers: fixed master per Vf/AR, random orientation, cubic box = 2L',fontsize=12)
    fig.tight_layout();fig.savefig(output/'fiber_scaling.png',dpi=180)
    fig.savefig(output/'fiber_scaling.svg');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,6.8))
    for row,n in enumerate((96,192)):
        for col,(key,title,fmt) in enumerate((('median_ms','Six-load time [ms]', '.0f'),
                ('iterations','Total CG iterations','d'),('correction_iterations','Physical correction iterations','d'))):
            z=np.array([[next(r[key] for r in agreed if r['N']==n and r['Vf_target']==vf and r['AR']==ar)
                for ar in (5,10,20)] for vf in (.05,.15,.25)])
            im=axs[row,col].imshow(z,cmap='viridis',aspect='auto')
            axs[row,col].set(xticks=range(3),xticklabels=['5','10','20'],yticks=range(3),
                yticklabels=['5%','15%','25%'],xlabel='Aspect ratio',ylabel='Fiber volume fraction',title=f'{n}³: {title}')
            for i in range(3):
                for j in range(3):axs[row,col].text(j,i,format(z[i,j],fmt if z.dtype.kind in 'iu' else '.0f'),
                    ha='center',va='center',color='white' if z[i,j]<(z.max()+z.min())/2 else 'black',fontsize=11)
            fig.colorbar(im,ax=axs[row,col],shrink=.82)
    fig.suptitle('Same voxel count within each row; physical domain and diameter resolution vary with AR',fontsize=11)
    fig.tight_layout();fig.savefig(output/'fiber_fixed_grid.png',dpi=180)
    fig.savefig(output/'fiber_fixed_grid.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    native=[r for r in records if r['native_resolution_6']]
    write_csv(output/'fiber_fixed_physical_resolution.csv',native)
    for vf in (.05,.15,.25):
        rs=sorted((r for r in native if r['Vf_target']==vf),key=lambda r:r['AR'])
        for ax,key in zip(axs,('median_ms','iterations')):
            ax.plot([r['AR'] for r in rs],[r[key]/(1000 if key=='median_ms' else 1) for r in rs],
                'o-',color=colors[vf],label=f'Vf {vf:.0%}')
            ax.grid(alpha=.18)
    axs[0].set(xlabel='Aspect ratio',ylabel='Six-load wall time [s]',title='Fixed 6 voxels / µm and 6 voxels / diameter')
    axs[1].set(xlabel='Aspect ratio',ylabel='Iterations over six loads',title='N = 60, 120, 150, 240 as AR increases')
    axs[0].legend();fig.tight_layout();fig.savefig(output/'fiber_fixed_resolution.png',dpi=180)
    fig.savefig(output/'fiber_fixed_resolution.svg');plt.close(fig)
