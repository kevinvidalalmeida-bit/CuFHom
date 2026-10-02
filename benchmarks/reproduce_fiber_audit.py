"""Rasterize saved continuous masters and optionally repeat the GPU audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from cufhom.pipeline.rve_generator import rasterize_continuous_fibers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    evidence=Path(__file__).resolve().parent/'performance_audit_20261001'
    p.add_argument('--saved',type=Path,default=evidence/'geometries')
    p.add_argument('--material',type=Path,default=evidence/'material.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grids',type=int,nargs='+',default=[32,48,64,96,128,192,256])
    p.add_argument('--no-native',action='store_true',help='Skip the extra 6 voxels/µm raster')
    p.add_argument('--run',action='store_true');p.add_argument('--repeats',type=int,default=9)
    a=p.parse_args();prepared=a.output/'fiber_geometries';prepared.mkdir(parents=True,exist_ok=True)
    cases=json.loads((a.saved/'cases.json').read_text());rasters=[]
    for case in cases:
        master=a.saved/(case['case']+'.csv')
        with master.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==case['master_sha256']
        table=pd.read_csv(master)
        for n in sorted(set(a.grids+([] if a.no_native else [round(6*case['box_um'])]))):
            geometry=prepared/case['case']/f'N{n}'
            result=rasterize_continuous_fibers(table,caja_um=case['box_um'],
                resolution=n/case['box_um'],output_dir=geometry)
            m=result['metadata']
            rasters.append(dict(case=case['case'],N=n,Vf_target=case['Vf_target'],AR=case['AR'],
                Vf_voxel=m['voxel_volume_fraction'],box_um=case['box_um'],length_um=case['length_um'],
                diameter_um=case['diameter_um'],voxels_per_um=n/case['box_um'],
                voxels_per_diameter=n*case['diameter_um']/case['box_um'],native_resolution_6=n==round(6*case['box_um']),
                master_sha256=case['master_sha256'],A2_voxel=m['A2_voxel'],
                binary_diameter_below_2_voxels=n*case['diameter_um']/case['box_um']<2,geometry=str(geometry)))
    manifest=prepared/'manifest.json';manifest.write_text(json.dumps(dict(cases=cases,rasters=rasters),indent=2)+'\n')
    if a.run:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_fiber_audit.py')),
            '--prepared',str(manifest),'--material',str(a.material),'--output',str(a.output),
            '--repeats',str(a.repeats)],check=True)
    print(manifest,flush=True)


if __name__=='__main__':main()
