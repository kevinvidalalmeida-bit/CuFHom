"""Hybrid 3D tangent action: cuFFT XY and cuFFTDx C2R/tangent/R2C along Z.

An isolated experiment, no padding or voxel changes, no production integration.
Compile cufftdx_tangent_candidate.cu into an external shared library first.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from profile_g09_fft import cp, ROOT
from profile_nonlinear_stages import parameters
from benchmark_cuda_graph_operator import sample
from pipeline.nonlinear_solver import _table, MixedProjector
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from ffthompy.tensors import fft
from cupyx.scipy.fft import get_fft_plan


class CufftdxTangentOperator:
    """Own XY buffers; return independently owned coordinate output."""
    def __init__(self, library, Q, gpu, response):
        if Q.shape[-1] != 240 or gpu.dtype != np.float32:
            raise ValueError('This compiled prototype requires 240 on Z and float32')
        self.Q, self.gpu, self.response = Q, gpu, response
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.candidate_info.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.candidate_info.restype = ctypes.c_int
        values = (ctypes.c_int*6)()
        self._check(self.library.candidate_info(values))
        self.info = dict(zip(('threads_x','threads_y','shared_bytes','storage_size',
                             'registers_per_thread','local_bytes_per_thread'), values))
        self.library.candidate_launch.argtypes = [ctypes.c_void_p]*7+[
            ctypes.c_float,ctypes.c_int,ctypes.c_size_t]
        self.library.candidate_launch.restype = ctypes.c_int
        self.xy = cp.empty((6,)+Q.hshape, dtype=cp.complex64)
        self.z_out = cp.empty_like(self.xy)
        self.h_out = cp.empty_like(self.xy)
        # Axes include the first dimension of each 3D component: 121 independent
        # XY complex transforms, interleaved along Z. No transpose or data copy.
        self.xy_plan = get_fft_plan(self.xy[0], axes=(0,1), value_type='C2C')

    @staticmethod
    def _check(code):
        if code:
            raise cp.cuda.runtime.CUDARuntimeError(code)

    def __call__(self, value):
        Q, gpu, r = self.Q, self.gpu, self.response
        h = Q._map(value, False)
        for c in range(6):
            self.xy_plan.fft(h[c], self.xy[c], cp.cuda.cufft.CUFFT_INVERSE)
        args = [a.data.ptr for a in (self.xy,self.z_out,r.direction,r.a,r.b,gpu.ids,gpu.table)]
        self._check(self.library.candidate_launch(*args,gpu.material.K,
            int(np.prod(Q.shape[:-1])),cp.cuda.get_current_stream().ptr))
        for c in range(6):
            self.xy_plan.fft(self.z_out[c], self.h_out[c], cp.cuda.cufft.CUFFT_FORWARD)
        return Q._map(self.h_out, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--profile-stages', action='store_true')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--inner', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.repeats,args.inner)<1:
        parser.error('New output file and positive repeats/inner required')
    p = parameters(); law = J2Material(**p['matrix'])
    phase, ori = p['phase_array'], p['ori_array']
    if args.small:
        phase = np.array(phase[:9,:13,:]); ori = np.array(ori[:9,:13,:])
        # Exercise fibers and their rotated off-diagonal stiffness on small test.
        phase[3:6,3:8,:] = 1
        ori[3:6,3:8,:, :] = np.array([1.,2.,3.])/np.sqrt(14)
        phase.flags.writeable = ori.flags.writeable = False
    shape = phase.shape
    ids, table = _table(phase,ori,law,p['fiber'])
    gpu = CuPyMaterial(cp,law,cp.asarray(ids),cp.asarray(table,dtype=cp.float32),np.float32)
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    fft.set_cupy_rfft_batch_mode('component')
    Q = DisplacementCoordinates(MixedProjector(shape,(1.,2.,3.),
                                [True,False,False,False,False,False],cp))
    rng = cp.random.RandomState(712)
    field = rng.standard_normal((6,)+shape,dtype=cp.float32)
    strain = np.float32(.003)*field; strain[0] += np.float32(.006)
    response = gpu.update(strain,initial_state(shape,cp,np.float32))
    candidate = CufftdxTangentOperator(args.library,Q,gpu,response)
    saved = [(a,a.copy()) for a in vars(response).values()] if args.small else []
    z = Q.adjoint(field)
    def original():
        return Q.adjoint(gpu.tangent(Q.strain(z),response))
    expected = original(); actual = candidate(z)
    error = float(cp.linalg.norm(actual-expected)/cp.linalg.norm(expected))
    scale = float(cp.max(cp.abs(expected)))
    maximum_error = float(cp.max(cp.abs(actual-expected)))/scale
    z2 = Q.adjoint(rng.standard_normal((6,)+shape,dtype=cp.float32))
    a2 = candidate(z2); ref2 = Q.adjoint(gpu.tangent(Q.strain(z2),response))
    second_error = float(cp.linalg.norm(a2-ref2)/cp.linalg.norm(ref2))
    vdot = lambda a,b: float(cp.sum((a.conj()*b).real,dtype=cp.float64))
    # Symmetry is a separate diagnostic; neither calculation modifies states.
    cross1, cross2 = vdot(z,a2), vdot(actual,z2)
    symmetry = abs(cross1-cross2)/max(np.sqrt(vdot(z,z)*vdot(a2,a2)),1e-30)
    positive = vdot(z,actual)>0
    for value, old in saved:
        cp.testing.assert_array_equal(value,old)
    checks = dict(relative_l2_error=error, scaled_max_error=maximum_error,
                  second_input_relative_error=second_error,symmetry_defect=symmetry,positive=positive)
    passed = bool(np.isfinite([error,maximum_error,second_error,symmetry]).all()
                  and max(error,second_error,symmetry)<2e-6 and maximum_error<1e-5 and positive)
    variants = dict(original=original,cufftdx=lambda:candidate(z))
    rows = {name:[] for name in variants}
    if passed:
        for fn in variants.values():
            fn()
        for rep in range(args.repeats):
            for name in (list(variants) if rep%2==0 else list(reversed(variants))):
                rows[name].append(sample(variants[name],args.inner))
    stages = None
    if passed and args.profile_stages:
        from profile_linear_stages import EventProfile
        profile = EventProfile()
        with ExitStack() as stack:
            stack.enter_context(patch.object(Q,'_map',profile.wrap('coordinate_map',Q._map)))
            stack.enter_context(patch.object(candidate,'xy_plan',SimpleNamespace(
                fft=profile.wrap('xy_fft',candidate.xy_plan.fft))))
            stack.enter_context(patch.object(candidate.library,'candidate_launch',
                profile.wrap('fused_z_tangent',candidate.library.candidate_launch)))
            for _ in range(5):
                candidate(z)
        cp.cuda.get_current_stream().synchronize()
        stages = dict(repeats=5, note='Separate instrumented calls, CUDA event intervals '
            'including launch gaps; not pure active hardware counters.',summary=profile.summary())
    report = dict(scope='Frozen full 3D Q* C Q action, actual G09 tables, synthetic '
        'plastic strain. cuFFT strided XY plus cuFFTDx fused Z. All FFT passes/maps '
        'and launches included. NOT a full nonlinear curve or CG solve.',
        shape=list(shape), small=args.small, seed=712, checks=checks, passed=passed,
        stages=stages,
        compiler='nvcc 13.0.88, -O3 -std=c++17 -arch=sm_120, no fast-math',
        mathdx='26.06.1 CUDA13 (external temporary directory)',kernel=candidate.info,
        library=str(args.library.resolve()),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in (Path(__file__),Path(__file__).with_name('cufftdx_tangent_candidate.cu'))},
        timings={name:dict(median_event_ms=float(np.median([r['event_ms'] for r in samples])),
                          median_wall_ms=float(np.median([r['wall_ms'] for r in samples])),
                          samples=samples) for name,samples in rows.items() if samples})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise SystemExit('Operator verification failed; do not use in a solver')


if __name__=='__main__':
    main()
