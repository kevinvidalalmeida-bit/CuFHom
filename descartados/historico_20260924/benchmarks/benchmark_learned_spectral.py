"""G09 fixed-SPD spectral pilot, training cost and held-out RHS evidence."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, cp, sync, quiet_solver,
    optimized_parameters, compile_cfield_geometry, solve_homogenization)
from linear_strategy_candidates import candidate
from learned_spectral_candidate import OperatorSamples, collect_samples, learned_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--blends', type=float, nargs='+', default=[.25,.5,1.])
    args = parser.parse_args()
    if args.output.exists() or args.repeats<2 or any(not 0<=b<=1 for b in args.blends):
        parser.error('New output directory, >=2 repeats and blends in [0,1] required')
    args.output.mkdir(parents=True)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    config = optimized_parameters(phase, ori, json.loads(MATERIAL.read_text()), compile_cfield_geometry(phase, ori))
    with quiet_solver():
        reference = solve_homogenization(config)
    samples = OperatorSamples()
    sync(); start = time.perf_counter()
    with collect_samples(samples), quiet_solver():
        solve_homogenization(dict(config, active_load_ids=[0,1,2], partial_load_output=True,
                                 solver_timing_path=str(args.output/'training.json')))
    model = samples.model(); sync(); training_s = time.perf_counter()-start
    (args.output/'model.json').write_text(json.dumps(model, indent=2)+'\n')
    print('training', training_s, model['pairs'], model['sampled_frequencies'], flush=True)
    paths = ['FFT/pipeline/fft_solver.py','FFT/pipeline/linear_coordinates.py','FFT/pipeline/linear_krylov.py',
             'FFT/benchmarks/learned_spectral_candidate.py','FFT/benchmarks/benchmark_learned_spectral.py']
    report = dict(training_s=training_s, training_loads=[0,1,2], held_out_loads=[3,4,5],
        scope='Same G09 geometry/material; held-out RHS, not unseen geometries. Training includes '
        'sampling, CPU fit and transfers after base warmup. Evaluation includes group/root setup '
        'within every six-load solve; fixed blocks per CG; no extra preconditioner FFT. '
        'Default physical tolerance 1e-5 and unchanged original-equation audits.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        variants={}, material=json.loads(MATERIAL.read_text()), shape=list(phase.shape), warmup_errors={})
    blends = {f'learned_{b:g}':b for b in args.blends}
    modes = ['production']+list(blends)
    def experiment(mode):
        return candidate('production') if mode=='production' else learned_candidate(model, blends[mode])
    for mode in modes:
        try:
            with experiment(mode), quiet_solver():
                solve_homogenization(config)
            print('warmup',mode,flush=True)
        except Exception as err:
            report['warmup_errors'][mode] = str(err)
            print('warmup error',mode,str(err),flush=True)
    for repeat in range(args.repeats):
        for mode in modes[repeat%len(modes):]+modes[:repeat%len(modes)]:
            entry = dict(repeat=repeat)
            sync(); start = time.perf_counter()
            try:
                with experiment(mode) as (infos, cache), quiet_solver():
                    result = solve_homogenization(dict(config,solver_timing_path=str(args.output/f'{repeat}_{mode}.json')))
                sync()
                error = float(np.linalg.norm(result-reference)/np.linalg.norm(reference))
                entry.update(wall_s=time.perf_counter()-start, Ceff=result.tolist(), difference=error, loads=infos,
                    cg=sum(i['coordinate_iterations']+i['correction_iterations'] for i in infos),
                    held_out_cg=sum(i['coordinate_iterations']+i['correction_iterations'] for i in infos[3:]),
                    passed=len(infos)==6 and np.isfinite(error) and error<=1e-6 and all(
                        i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos))
            except Exception as err:
                sync(); entry.update(wall_s=time.perf_counter()-start,error=str(err),passed=False)
            report['variants'].setdefault(mode,[]).append(entry)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(repeat,mode,entry['wall_s'],entry.get('cg'),entry.get('held_out_cg'),entry.get('error',''),flush=True)
    report['summary'] = {mode:dict(median_s=float(np.median([r['wall_s'] for r in runs])),
        passed=all(r['passed'] for r in runs)) for mode,runs in report['variants'].items()}
    base = report['summary']['production']['median_s']
    for mode, summary in report['summary'].items():
        saving = base-summary['median_s']
        summary['training_amortization_solves'] = training_s/saving if saving>0 and summary['passed'] else None
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
