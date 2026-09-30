"""Stage microbenchmarks on the actual 240^3 geometry; not full-curve speedups."""
import argparse
import json
from pathlib import Path
import numpy as np
from profile_g09_fft import cp, sync
from profile_nonlinear_stages import parameters
from pipeline.nonlinear_solver import _table, MixedProjector
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from ffthompy.tensors import fft
from hotspot_candidates import tangent_registers, coordinate_specialized


def timed(fn, repeats=7, inner=10):
    fn(); sync()
    samples = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(inner):
            result = fn()
        end.record(); end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, end)/inner)
        del result
    return dict(median_ms=float(np.median(samples)), samples_ms=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    p = parameters(); shape = p['phase_array'].shape
    material = J2Material(**p['matrix'])
    ids, table = _table(p['phase_array'], p['ori_array'], material, p['fiber'])
    gpu = CuPyMaterial(cp, material, cp.asarray(ids), cp.asarray(table, dtype=cp.float32), np.float32)
    rng = cp.random.RandomState(451)
    field = rng.standard_normal((6,)+shape, dtype=cp.float32)
    strain = field*np.float32(.003)
    strain[0] += np.float32(.006)
    response = gpu.update(strain, initial_state(shape, cp, np.float32))
    report = dict(shape=shape, scope='GPU stages, actual G09 orientation/phase tables; synthetic '
        'strain/direction. Warm kernels/plans. Not a full-curve speedup claim.', stages={})
    reference = gpu.tangent(field, response)
    candidate = tangent_registers(gpu, field, response)
    error = float(cp.linalg.norm(candidate-reference)/cp.linalg.norm(reference))
    if error > 2e-6:
        raise RuntimeError('Tangent mismatch')
    report['tangent_relative_error'] = error
    del reference, candidate, strain
    report['stages']['tangent_original'] = timed(lambda: gpu.tangent(field, response))
    report['stages']['tangent_registers'] = timed(lambda: tangent_registers(gpu, field, response))
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    fft.set_cupy_rfft_batch_mode('full')
    P = MixedProjector(shape, (1.,1.,1.), [True,False,False,False,False,False], cp)
    Q = DisplacementCoordinates(P)
    h = fft.rfftn_cached(field, shape, (1,2,3)); z = Q._map(h, True)
    for adjoint,x in [(True,h), (False,z)]:
        label = 'adjoint' if adjoint else 'strain'
        original = Q._map(x, adjoint); candidate = coordinate_specialized(Q, x, adjoint)
        error = float(cp.linalg.norm(candidate-original)/cp.linalg.norm(original))
        if error > 2e-6:
            raise RuntimeError('Coordinate mismatch')
        report[label+'_relative_error'] = error
        report['stages'][label+'_original'] = timed(lambda: Q._map(x, adjoint))
        report['stages'][label+'_specialized'] = timed(lambda: coordinate_specialized(Q, x, adjoint))
    del h,z,original,candidate,response
    # Batched raw cuFFT execution versus the existing cached pair. Each inverse
    # consumes a fresh forward result: C2R may overwrite its complex input.
    def existing_pair():
        h = fft.rfftn_cached(field, shape, (1,2,3))
        return fft.irfftn_unscaled_cached(h, shape, (1,2,3))
    report['stages']['fft_cached_pair'] = timed(existing_pair)
    for batch in (1,2,3,6):
        complex_output = cp.empty((6,)+shape[:-1]+(shape[-1]//2+1,), dtype=cp.complex64)
        real_output = cp.empty_like(field)
        forward = fft._maybe_cupy_plan(field[:batch], (1,2,3), 'R2C')
        inverse = fft._maybe_cupy_plan(complex_output[:batch], (1,2,3), 'C2R', shape=shape)
        def pair():
            for begin in range(0,6,batch):
                forward.fft(field[begin:begin+batch], complex_output[begin:begin+batch], cp.cuda.cufft.CUFFT_FORWARD)
                inverse.fft(complex_output[begin:begin+batch], real_output[begin:begin+batch], cp.cuda.cufft.CUFFT_INVERSE)
            return real_output
        out = pair()
        error = float(cp.linalg.norm(out/np.prod(shape)-field)/cp.linalg.norm(field))
        if error > 2e-6:
            raise RuntimeError('FFT mismatch')
        report['stages'][f'fft_raw_batch{batch}'] = dict(timed(pair), relative_error=error)
        del complex_output, real_output, out
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
