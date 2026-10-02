"""Store lightweight measured evidence and verify frozen source/input hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];records={}
    for path in sorted(a.input.rglob('report.json')):
        records[str(path.relative_to(a.input))]=json.loads(path.read_text())
    for pattern in ('*.json','*/current_0.json','*/dual_metric_0.json','*/dual_strict_half_rtol_0.json',
            '*/auto_0.json','*/component_0.json','*/full_0.json','fiber_geometries/*/periodic_repair.json'):
        for path in a.input.glob(pattern):records[str(path.relative_to(a.input))]=json.loads(path.read_text())
    manifest=json.loads((a.input/'fiber_geometries/manifest.json').read_text())
    records['fiber_geometries/manifest.json']=manifest
    driver=Path('/proc/driver/nvidia/params')
    records['profiling_permission']=dict(driver_flag=[line for line in driver.read_text().splitlines()
        if 'Profiling' in line] if driver.exists() else [],
        probe_logs={path.name:path.read_text() for path in Path('/tmp').glob('cufhom-ncu-probe*.log')})
    # Known fixture failure: keep it explicit instead of excluding an unsuccessful experiment.
    records['operator_G09_rhs6_failure']=dict(status='out of memory in diagnostic fixture',
        allocated_bytes=14720575488,attempted_extra_bytes=1990656000,
        reason='Serial, batched, graph and norm-validation buffers coexisted; no completed timing report.',
        conclusion='Not an inherent limit of a compact six-RHS solver; rerun only after reducing fixture memory.')
    for name in ('acoustic_local_validation_20261001','physical_metric_validation_20261001'):
        folder=a.input.parent/name
        for path in folder.glob('*.json'):records['validation/'+name+'/'+path.name]=json.loads(path.read_text())
    reference=records['G09/report.json']['source_sha256']
    assert all(sha(root/path)==value for path,value in reference.items()),'Production source changed'
    measured_baselines=[r for name,r in records.items() if name.endswith('/report.json') and
        'statistics_s' in r and 'source_sha256' in r and r.get('runs')]
    assert all(r['source_sha256']==reference for r in measured_baselines)
    input_hashes=dict(manifest['source_sha256'])
    for r in measured_baselines:input_hashes.update(r.get('input_sha256',{}))
    # profile_g09_linear uses named input hashes; only the audit harness uses path keys.
    checked={str(path):value for path,value in input_hashes.items() if Path(path).is_file()}
    assert all(sha(Path(path))==value for path,value in checked.items()),'An original/input file changed'
    bundle=a.output/'evidence.json.gz'
    encoded=json.dumps(records,separators=(',',':'),allow_nan=False).encode()
    with bundle.open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(encoded)
    scripts={str(path.relative_to(root)):sha(path) for path in (root/'benchmarks').glob('*.py')
        if path.name.startswith(('audit_','build_','prepare_fiber','profile_linear','run_audit','run_fiber','reproduce_fiber','write_performance','archive_audit'))}
    archive=root/'descartados/performance_audit_20261001'
    scripts.update({str(path.relative_to(root)):sha(path) for path in archive.glob('*.py')})
    hashes=sorted(set(r['benchmark_sha256'] for r in measured_baselines))
    result=dict(production_commit='92c9ac5b8204288244e82536b254078d6d767e66',production_sha256=reference,
        production_unchanged=True,original_input_files_verified=len(checked),input_sha256=checked,
        evidence_sha256=sha(bundle),evidence_records=len(records),current_harness_sha256=scripts,
        benchmark_hashes_as_measured=hashes,
        note='Harness evolved to add geometry families, NVTX labels and an explicit FFT policy; all recorded production hashes agree. '
            'Baselines used auto policy. Diagnostic/variant timing is kept separate. Individual report hashes remain unchanged in the bundle.',
        counters_status='ERR_NVGPUCTRPERM; RmProfilingAdminOnly=1',
        reproduction_N32=dict(masters=12,arrays=24,phase_and_orientation_sha256_identical=True))
    (a.output/'evidence_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    archive_manifest=dict(scope='Isolated pilots, excluded from installed package and production imports',
        files={str(path.relative_to(root)):sha(path) for path in archive.iterdir() if path.is_file()},
        evidence='benchmarks/performance_audit_20261001/evidence_manifest.json')
    (archive.parent/'manifest_performance_audit_20261001.json').write_text(json.dumps(archive_manifest,indent=2)+'\n')
    print('Evidence:',len(records),'records,',bundle.stat().st_size,'bytes; verified original/input files:',len(checked))


if __name__=='__main__':main()
