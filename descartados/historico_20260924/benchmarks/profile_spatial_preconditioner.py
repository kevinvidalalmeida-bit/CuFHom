"""Isolate setup/correction costs on a real G09 Newton system (no defaults)."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from profile_nonlinear_stages import parameters, solver, sync
from profile_linear_stages import EventProfile
from benchmark_spatial_preconditioner import timed
from nonlinear_lowlevel_candidates import action_context
from spatial_preconditioner_candidate import BlockCompliance, cp, device_dot
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction
from ffthompy.tensors import fft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use new file')
    original = FusedCoordinatePCG.solve
    result = {}
    def solve(engine, action, rhs, dot, tolerance, maxiter, reference):
        if result:
            return original(engine, action, rhs, dot, tolerance, maxiter, reference)
        fields, gpu = action_context(action)
        pc = BlockCompliance(engine.Q, gpu, fields['response'], edge=8, support='matrix')
        reduction = CupyWarpCGReduction(SimpleNamespace(N=engine.Q.shape, val=rhs.real), max_blocks=16384)
        def warp_dot():
            reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                (rhs.view(cp.float32), rhs.view(cp.float32), reduction.partial, np.int64(rhs.size*2)))
            return cp.sum(reduction.partial, dtype=cp.float64)
        def repeated(fn, n=5):
            for _ in range(n):
                fn()
        # Warm both paths before recording timing.
        repeated(lambda: pc.correction(rhs), 2)
        repeated(lambda: device_dot(rhs, rhs), 2)
        repeated(warp_dot, 2)
        result['plain_before_seconds_per_call'] = {
            'correction': timed(lambda: repeated(lambda: pc.correction(rhs)))[1]/5,
            'generic_dot': timed(lambda: repeated(lambda: device_dot(rhs, rhs)))[1]/5,
            'warp_dot': timed(lambda: repeated(warp_dot))[1]/5}
        profiler = EventProfile()
        with ExitStack() as stack:
            for obj, name in ((pc, 'correction'), (pc, 'local_action'),
                (pc, 'restrict_kernel'), (pc, 'prolong_kernel'),
                (engine.Q, 'strain'), (engine.Q, 'adjoint'), (engine.Q, '_map'),
                (fft, 'rfftn_cached'), (fft, 'irfftn_unscaled_cached')):
                stack.enter_context(patch.object(obj, name, profiler.wrap(name, getattr(obj, name))))
            repeated(lambda: pc.correction(rhs))
        result['stages_five_calls'] = profiler.summary()
        result['plain_after_seconds_per_call'] = {
            'correction': timed(lambda: repeated(lambda: pc.correction(rhs)))[1]/5,
            'generic_dot': timed(lambda: repeated(lambda: device_dot(rhs, rhs)))[1]/5,
            'warp_dot': timed(lambda: repeated(warp_dot))[1]/5}
        result['dot_relative_difference'] = abs(float(warp_dot())-float(device_dot(rhs,rhs)))/float(warp_dot())
        source = Path(__file__).with_name('spatial_preconditioner_candidate.py')
        result['candidate_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        result['scope'] = 'First elastic Newton system of actual G09 240^3; intervals include launch gaps; '
        result['scope'] += 'not a full-curve timing. No simultaneous GPU benchmark.'
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
        return original(engine, action, rhs, dot, tolerance, maxiter, reference)
    p = parameters(); p.update(path=[0., .00025], coordinate_pcg='fused',
        reuse_trial_response=True, inner_forcing='ew_target', cupy_rfft_batch_mode='auto')
    with patch.object(FusedCoordinatePCG, 'solve', solve):
        solver.solve_nonlinear_homogenization(p)


if __name__ == '__main__':
    main()
