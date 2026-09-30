"""Diagnostic only: persistent buffers/graphs for a frozen nonlinear tangent.

No solver defaults are changed. The graph owns its buffers and frozen response;
its output is borrowed and overwritten on the next execution. This is NOT a
drop-in replacement for an operator returning independently owned arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from profile_g09_fft import cp, ROOT, G09_N240, MATERIAL
from profile_nonlinear_stages import parameters
from pipeline.nonlinear_solver import _table, MixedProjector
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state
from pipeline.nonlinear_coordinates import DisplacementCoordinates, _CODE
from ffthompy.tensors import fft


class FrozenTangentGraph:
    """One fixed Q* C Q action, no PTC shift and no state updates or CG loop."""
    def __init__(self, coordinates, material, response, value):
        self.Q, self.material, self.response = coordinates, material, response
        Q = coordinates
        if type(Q) is not DisplacementCoordinates or Q.xp is not cp:
            raise ValueError('Diagnostic supports GPU displacement coordinates only')
        if np.prod(Q.shape) % 2 or value.dtype not in (cp.complex64, cp.complex128):
            raise ValueError('Even volume and complex64/128 required by this diagnostic')
        if value.shape != (Q.size,) or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous coordinate vector')
        d = value.real.dtype.type
        self.input = value.copy()
        self.output = cp.empty_like(value)
        self.hstrain = cp.empty((6,)+Q.hshape, dtype=value.dtype)
        self.hstress = cp.empty_like(self.hstrain)
        self.strain = cp.empty((6,)+Q.shape, dtype=d)
        self.stress = cp.empty_like(self.strain)
        name = 'float' if d == np.float32 else 'double'
        code = _CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
        self.map_kernel = cp.RawKernel(code, 'coordinates')
        self.map_args = (*(np.int64(n) for n in Q.shape), np.int64(Q.hshape[-1]),
                         *(d(1/L) for L in Q.P.lengths), d(Q.norm),
                         np.int32(sum(1<<i for i in Q.P.free)))
        self.forward = fft._maybe_cupy_plan(self.stress[:1], (1,2,3), 'R2C')
        self.inverse = fft._maybe_cupy_plan(self.hstrain[:1], (1,2,3), 'C2R', shape=Q.shape)
        # Hold each view and both plan workspaces alive through capture/replay.
        self.views = tuple(tuple(a[c:c+1] for c in range(6)) for a in
                           (self.hstrain, self.strain, self.stress, self.hstress))
        self.tangent_args = (self.strain, response.direction, response.a, response.b,
                            material.ids, material.table, self.stress,
                            np.int64(material.n), d(material.material.K))
        self.scratch_bytes = sum(a.nbytes for a in (self.input, self.output,
                                self.hstrain, self.hstress, self.strain, self.stress))
        self.run_raw()
        cp.cuda.get_current_stream().synchronize()
        capture_stream = cp.cuda.Stream(non_blocking=True)
        with capture_stream:
            self.run_raw()  # compile/plan/first execution outside capture
        capture_stream.synchronize()
        started = time.perf_counter()
        with capture_stream:
            capture_stream.begin_capture()
            self.run_raw()
            self.graph = capture_stream.end_capture()
            self.graph.launch(stream=capture_stream)
        capture_stream.synchronize()
        self.capture_and_first_launch_s = time.perf_counter()-started

    def run_raw(self):
        Q = self.Q
        self.map_kernel(((Q.count+255)//256,), (256,),
                        (self.input, self.hstrain, *self.map_args, np.int32(0)))
        hs, strain, stress, ht = self.views
        for c in range(6):
            # C2R consumes hstrain; the map above refills it on EVERY execution.
            self.inverse.fft(hs[c], strain[c], cp.cuda.cufft.CUFFT_INVERSE)
        self.material.tangent_kernel(((self.material.n+255)//256,), (256,), self.tangent_args)
        for c in range(6):
            self.forward.fft(stress[c], ht[c], cp.cuda.cufft.CUFFT_FORWARD)
        self.map_kernel(((Q.count+255)//256,), (256,),
                        (self.hstress, self.output, *self.map_args, np.int32(1)))
        return self.output

    def run_graph(self):
        self.graph.launch(stream=cp.cuda.get_current_stream())
        return self.output

    def run_graph_with_copy(self, value):
        if value.shape != self.input.shape or value.dtype != self.input.dtype:
            raise ValueError('Input cannot change shape or precision')
        cp.copyto(self.input, value)
        return self.run_graph()


def sample(fn, inner):
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    started = time.perf_counter()
    start.record()
    for _ in range(inner):
        out = fn()
    end.record(); end.synchronize()
    return dict(event_ms=cp.cuda.get_elapsed_time(start, end)/inner,
                wall_ms=(time.perf_counter()-started)*1000/inner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--inner', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.inner < 1:
        parser.error('Use a new output path and positive repetition counts')
    p = parameters(); shape = p['phase_array'].shape
    law = J2Material(**p['matrix'])
    ids, table = _table(p['phase_array'], p['ori_array'], law, p['fiber'])
    gpu = CuPyMaterial(cp, law, cp.asarray(ids), cp.asarray(table, dtype=cp.float32), np.float32)
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    fft.set_cupy_rfft_batch_mode('component')
    rng = cp.random.RandomState(701)
    strain = rng.standard_normal((6,)+shape, dtype=cp.float32)*np.float32(.003)
    strain[0] += np.float32(.006)
    response = gpu.update(strain, initial_state(shape, cp, np.float32))
    P = MixedProjector(shape, (1.,1.,1.), [True,False,False,False,False,False], cp)
    Q = DisplacementCoordinates(P)
    z = Q.adjoint(strain)
    del strain
    original = lambda: Q.adjoint(gpu.tangent(Q.strain(z), response))
    candidate = FrozenTangentGraph(Q, gpu, response, z)
    reference = original()
    variants = dict(original=original, persistent=candidate.run_raw,
                    graph_fixed_input=candidate.run_graph,
                    graph_with_input_copy=lambda: candidate.run_graph_with_copy(z))
    checks = {}
    for name, fn in variants.items():
        out = fn()
        error = float(cp.linalg.norm(out-reference)/cp.linalg.norm(reference))
        checks[name] = dict(relative_l2_error=error,
                            bitwise_equal=bool(cp.array_equal(out, reference)))
        if not np.isfinite(error) or error > 2e-6:
            raise RuntimeError(f'{name} mismatched the frozen tangent action')
    # Replay must pick up changed input, not just return the first captured action.
    z2 = z*np.float32(1.375)
    expected2 = Q.adjoint(gpu.tangent(Q.strain(z2), response))
    out2 = candidate.run_graph_with_copy(z2)
    checks['changed_input'] = dict(relative_l2_error=float(
        cp.linalg.norm(out2-expected2)/cp.linalg.norm(expected2)))
    if checks['changed_input']['relative_l2_error'] > 2e-6:
        raise RuntimeError('Graph did not use the new input')
    cp.copyto(candidate.input, z)
    del reference, z2, expected2, out2
    samples = {name: [] for name in variants}
    names = list(variants)
    for rep in range(args.repeats):
        order = names[rep % len(names):]+names[:rep % len(names)]
        for name in order:
            samples[name].append(sample(variants[name], args.inner))
    timings = {name: dict(median_event_ms=float(np.median([r['event_ms'] for r in rows])),
                         median_wall_ms=float(np.median([r['wall_ms'] for r in rows])),
                         samples=rows) for name, rows in samples.items()}
    paths = [Path(__file__), Path(fft.__file__),
             ROOT/'FFT/pipeline/nonlinear_coordinates.py', ROOT/'FFT/pipeline/nonlinear_material.py']
    report = dict(scope='Diagnostic frozen Q* C Q, actual G09 240^3 phase/orientation '
        'table, SYNTHETIC strain, NOT a CG solve or full-curve speedup. No defaults changed. '
        'Rotated timings, warmed plans, copied-input variant includes the device copy. '
        'Graph output is borrowed; original output has independent ownership.',
        shape=list(shape), repeats=args.repeats, inner=args.inner, seed=701,
        matrix=p['matrix'], fiber=p['fiber'], geometry=str(G09_N240), material_file=str(MATERIAL),
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        capture_and_first_launch_s=candidate.capture_and_first_launch_s,
        persistent_scratch_bytes=candidate.scratch_bytes,
        memory_note='Scratch excludes response, input z, cuFFT plan workspaces and memory-pool reserve.',
        checks=checks, timings=timings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
