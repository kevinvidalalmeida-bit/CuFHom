"""Prepare short-fiber Vf/AR sweeps with box=2*axis length; no solver edits.

Retain each continuous master across resolutions. Existing masters are copied,
never modified. Missing Vf/AR corners and three AR=10 cases are prepared
locally. Capsule geometry and binary voxelization match the current generator.
"""
import argparse,json,hashlib,time,importlib.util,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import numba
from cufhom.pipeline.rve_generator import generate_rve_main,rasterize_continuous_fibers

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--audit-module',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grids',type=int,nargs='+',default=[32,48,64,96,128,192,256]);p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);numba.set_num_threads(a.threads)
    spec=importlib.util.spec_from_file_location('periodic_audit',a.audit_module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=json.loads(a.config.read_text());cases=[];hashes={};gen=cfg['geometry_generation']
    for gid in (0,1,2,3,4,8,9):
        source=a.campaign/f'geometry_{gid:02d}';info=json.loads((source/'generation_result.json').read_text())
        assert info['accepted_local']
        row=info['design'];vf=float(row['Vf_target']);ar=float(row['aspect_ratio'])
        name=f'sfrp_vf{int(round(vf*100)):02d}_ar{ar:g}';folder=a.output/name;folder.mkdir(exist_ok=True)
        table=folder/'continuous_fibers_master.csv';original=source/'continuous_fibers_master.csv'
        hashes[str(original)]=sha(original)
        if not table.exists():shutil.copyfile(original,table)
        cases.append(dict(case=name,Vf_target=vf,AR=ar,diameter_um=float(row['d_um']),length_um=float(row['L_um']),
            box_um=float(row['box_um']),source=str(source),source_type='existing campaign master',table=str(table),
            seed=int(row['seed']),source_generation_acceptance=info['acceptance']))
    # Missing dilute-long corner: a random, orientation-balanced subset of the
    # existing valid dense-long master. Removing fibers cannot create contacts.
    dense=next(c for c in cases if c['Vf_target']==.25 and c['AR']==20.)
    name='sfrp_vf05_ar20';folder=a.output/name;folder.mkdir(exist_ok=True);table=folder/'continuous_fibers_master.csv'
    master=pd.read_csv(dense['table']);cap=np.pi*20/4+np.pi/6;count=int(np.floor(.05*40**3/cap+.5))
    if not table.exists():
        rng=np.random.default_rng(2026100105);best=None;score=float('inf')
        dirs=master[['ux','uy','uz']].to_numpy()
        for _ in range(200):
            ix=np.sort(rng.choice(len(master),count,replace=False));A2=dirs[ix].T@dirs[ix]/count
            err=np.linalg.norm(A2-np.eye(3)/3)
            if err<score:best=ix;score=err
        master.iloc[best].to_csv(table,index=False)
    cases.append(dict(case=name,Vf_target=.05,AR=20.,diameter_um=1.,length_um=20.,box_um=40.,
        source=dense['table'],source_type='orientation-balanced random subset of dense-long master',table=str(table),seed=2026100105))
    # Generate missing continuous RVEs using the existing CPU generator.
    for vf,ar,seed in ((.25,5.,2026100125),(.15,10.,2026100115),(.05,10.,2026100105),(.25,10.,202610012510)):
        name=f'sfrp_vf{int(vf*100):02d}_ar{ar:g}';folder=a.output/name;folder.mkdir(exist_ok=True)
        table=folder/'continuous_fibers_master.csv';request=dict(L_um=ar,d_um=1.,gap_um=0.,caja_um=2*ar,
            resol=6.,Vf_target=vf,seed=seed,output_dir=str(folder),a11=1/3,a22=1/3,
            compute_metrics=False,export_fiber_table=True,save_continuous_geometry=True,num_cores=a.threads,
            sam_num_threads=a.threads,sam_geometry_backend='numba',sam_verbose=False,
            **{k:v for k,v in gen.items() if k.startswith('sam_') and k!='sam_cupy_min_candidate_pairs'})
        (folder/'generation_request.json').write_text(json.dumps(request,indent=2)+'\n')
        if not table.exists():
            t=time.perf_counter();info=generate_rve_main(request)
            (folder/'generation_result.json').write_text(json.dumps(info,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
            print(name,'generated in',time.perf_counter()-t,'s',flush=True)
            if not (info.get('sam_vf_ok') and info.get('sam_overlap_ok')):raise RuntimeError(f'Generator rejected {name}')
        cases.append(dict(case=name,Vf_target=vf,AR=ar,diameter_um=1.,length_um=ar,box_um=2*ar,
            source_type='new local continuous RVE using unchanged generator',table=str(table),seed=seed))
    rows=[]
    for case in cases:
        table=pd.read_csv(case['table']);fibers=table[['cx_um','cy_um','cz_um','ux','uy','uz','L_um','d_um']].to_numpy()
        assert abs(case['box_um']/case['length_um']-2.)<1e-12
        t=time.perf_counter();audit=module.audit_periodic_fibers(fibers,case['box_um'],1/120)
        original_audit=audit;removed=[];original_indices=list(range(len(table)))
        while not audit['accepted']:
            # Repair only the new copy, by deleting one participant of the
            # worst forbidden contact. Choose the deletion with smaller A2 error.
            folder=a.output/case['case'];backup=folder/'source_master_before_repair.csv'
            if not backup.exists():shutil.copyfile(case['table'],backup)
            i,j=audit['worst']['i'],audit['worst']['j'];options=[]
            for drop in sorted(set((i,j))):
                candidate=np.delete(fibers[:,3:6],drop,axis=0)
                options.append((np.linalg.norm(candidate.T@candidate/len(candidate)-np.eye(3)/3),drop))
            drop=min(options)[1];removed.append(original_indices.pop(drop));table=table.drop(table.index[drop]).reset_index(drop=True)
            fibers=table[['cx_um','cy_um','cz_um','ux','uy','uz','L_um','d_um']].to_numpy()
            audit=module.audit_periodic_fibers(fibers,case['box_um'],1/120)
        if removed:
            table.to_csv(case['table'],index=False)
            case['periodic_repair']=dict(initial_audit=original_audit,removed_original_rows=removed,
                scope='Local master only; minimum-contact repair by deleting offending fibers; original source unchanged.')
            (a.output/case['case']/'periodic_repair.json').write_text(json.dumps(case['periodic_repair'],indent=2)+'\n')
            print(case['case'],'periodic repair removed',len(removed),'fibers',flush=True)
        elif (a.output/case['case']/'periodic_repair.json').exists():
            case['periodic_repair']=json.loads((a.output/case['case']/'periodic_repair.json').read_text())
        case['periodic_audit']=audit;case['audit_wall_s']=time.perf_counter()-t
        case['fiber_count']=len(table);case['master_sha256']=sha(Path(case['table']))
        dirs=fibers[:,3:6];case['A2_continuous']=(dirs.T@dirs/len(dirs)).tolist()
        case['continuous_capsule_vf']=float(len(table)*(np.pi*case['diameter_um']**2*case['length_um']/4+
            np.pi*case['diameter_um']**3/6)/case['box_um']**3)
        if abs(case['continuous_capsule_vf']-case['Vf_target'])>.006:raise RuntimeError('Repair exceeds historical Vf tolerance')
        if not audit['accepted']:raise RuntimeError(f'Periodic audit failed: {case["case"]}: {audit}')
        (a.output/case['case']/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        for n in sorted(set(a.grids+[int(round(6*case['box_um']))])):
            folder=a.output/case['case']/f'N{n}'
            t=time.perf_counter()
            if not (folder/'raster_manifest.json').exists():
                meta=rasterize_continuous_fibers(table,caja_um=case['box_um'],resolution=n/case['box_um'],output_dir=folder)['metadata']
            else:meta=json.loads((folder/'raster_manifest.json').read_text())
            row=dict(case=case['case'],N=n,Vf_target=case['Vf_target'],Vf_voxel=meta['voxel_volume_fraction'],AR=case['AR'],
                box_um=case['box_um'],length_um=case['length_um'],diameter_um=case['diameter_um'],
                voxels_per_um=n/case['box_um'],voxels_per_diameter=n*case['diameter_um']/case['box_um'],
                native_resolution_6=n==round(6*case['box_um']),raster_s=time.perf_counter()-t,
                geometry=str(folder),master_sha256=case['master_sha256'],A2_voxel=meta['A2_voxel'],
                binary_diameter_below_2_voxels=n*case['diameter_um']/case['box_um']<2.)
            rows.append(row);print(case['case'],n,'Vf',row['Vf_voxel'],'raster_s',row['raster_s'],flush=True)
        (a.output/'manifest.json').write_text(json.dumps(dict(cases=cases,rasters=rows,source_sha256=hashes,
            scope='Short fibers; capsule axis L, d=1um; cubic periodic box=2L; continuous masters reused at each N. '
            'Historical relative contact tolerance=1/120; source generator and solver are unchanged.'),indent=2)+'\n')
    assert hashes=={p:sha(Path(p)) for p in hashes}
if __name__=='__main__':main()
