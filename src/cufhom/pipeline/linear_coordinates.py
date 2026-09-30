"""Isometric displacement PCG adapter for the existing linear FFT problem.

Uses the same Q mapping as the nonlinear solver, with all mean fluctuations
fixed to zero. No frequencies, voxels or material coefficients are removed.
"""
import numpy as np
from cufhom_core.general.solver import _finalize_cg_info
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction
from .nonlinear_solver import MixedProjector, _cg, cp
from .nonlinear_coordinates import DisplacementCoordinates
from .nonlinear_reference import IsotropicReference
from .linear_krylov import SymmetricCoordinates, fused_cg


def make_linear_solver(*, A, GN, pb, fallback, bulk, shear, preconditioning=None):
    """Factory consumed by the core's optional per-load Krylov adapter."""
    mode = preconditioning or pb.solver.get('linear_preconditioning', 'symmetric_fused')
    if mode not in ('matrix', 'symmetric_fused'):
        raise ValueError('Unknown linear preconditioning mode')
    def solve(solver, Afun, B, x0, par, callback):
        if callback is not None:
            raise ValueError('Displacement PCG does not support callbacks')
        shape = tuple(B.N)
        volume = int(np.prod(shape))
        dtype = B.val.dtype
        P = MixedProjector(shape, tuple(pb.Y), np.ones(6, bool), cp)
        if mode == 'symmetric_fused':
            Q = SymmetricCoordinates(P, bulk, shear)
            inner_cg, precondition = fused_cg, None
        else:
            Q = DisplacementCoordinates(P)
            reference = IsotropicReference(P, bulk, shear)
            reference.set_shift(0., dtype)
            inner_cg, precondition = _cg, lambda v: Q.precondition(v, reference)
        reduction = CupyWarpCGReduction(B, max_blocks=16384)

        def dot(a, b):
            reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                (a.view(dtype), b.view(dtype), reduction.partial, np.int64(a.view(dtype).size)))
            return float(cp.sum(reduction.partial, dtype=cp.float64))

        def action(z):
            return Q.adjoint(A(B.copy(val=Q.strain(z))).val)

        def action_and_dot(z, *, direction=None):
            strain = (Q.strain(z) if direction is None
                      else Q.strain_updated_direction(z, *direction))
            return Q.adjoint_and_dot(A(B.copy(val=strain)).val, z)

        fused_options = {}
        if mode == 'symmetric_fused' and pb.solver.get('linear_kernel_fusion', True):
            fused_options['action_and_dot'] = action_and_dot

        # The existing direct-projection raw-C2R operator multiplies both
        # sides by volume. Remove that common factor in physical coordinates.
        rhs_norm = np.sqrt(dot(B.val, B.val)/volume)
        threshold = max(par.get('atol', 0.), par['rtol']*rhs_norm)
        relative = threshold/rhs_norm if rhs_norm else par['rtol']
        initial, initial_norm = None, None
        if x0 is not None:
            if not bool(cp.all(cp.isfinite(x0.val))):
                raise ValueError('initial_solution_fields must be finite')
            # Snapshots are physical fluctuation strains, not the current
            # material's whitened coordinates. Project with G, NOT QR RQ*.
            # This also removes means/Nyquist modes and owns a new buffer.
            trial = B.copy(val=P(x0.val))
            residual = B.val-Afun(trial).val
            initial_norm = np.sqrt(dot(residual, residual)/volume)
            if not np.isfinite(initial_norm):
                raise ValueError('Nonfinite initial-solution residual')
            # A distant material can give a worse guess than zero. Keep the
            # same sample order and fall back to a cold solve in that case.
            if initial_norm <= max(rhs_norm, threshold):
                initial = trial
            else:
                del residual
            del trial

        if initial is not None and initial_norm <= threshold:
            result, iterations, norm = initial, 0, initial_norm
        else:
            if initial is not None:
                rhs = Q.adjoint(residual)/dtype.type(volume)
                warm_squared = dot(rhs, rhs)
                # Solve A_new * delta = b_new - A_new * initial, starting
                # delta at zero. The stopping scale belongs to the ORIGINAL
                # RHS, never to the smaller correction RHS.
                # ||r_physical||_RMS <= sqrt(N)*||R^-1||*||r_coordinates||.
                # Use this bound to avoid an unnecessary strain-CG cleanup
                # when the correction has a different spectral distribution.
                inverse_root_norm = (np.sqrt(max(2*shear, bulk+4*shear/3))
                                     if mode == 'symmetric_fused' else 1.)
                if warm_squared > 0:
                    relative = threshold/(np.sqrt(volume*warm_squared)*inverse_root_norm)
                del residual
            else:
                rhs = Q.adjoint(B.val)/dtype.type(volume)
            z, iterations = inner_cg(action, rhs, dot, relative, par['maxiter'], cp,
                                     fused=True, precondition=precondition, **fused_options)
            result = B.copy(val=Q.strain(z))
            if initial is not None:
                result.val += initial.val
            residual = B.val-Afun(result).val
            norm = np.sqrt(dot(residual, residual)/volume)
        correction_iterations = 0
        if not np.isfinite(norm) or norm > threshold:
            remaining = par['maxiter']-iterations
            if remaining <= 0:
                raise RuntimeError('Displacement PCG exhausted its iteration budget before audit convergence')
            result, correction = fallback(solver=solver, Afun=Afun, B=B, x0=result,
                par=dict(par, rtol=par['rtol']*.5, atol=par.get('atol', 0.)*.5,
                         maxiter=remaining, keep_solution_on_device=True), callback=None)
            correction_iterations = correction['kit']
            residual = B.val-Afun(result).val
            norm = np.sqrt(dot(residual, residual)/volume)
        if not np.isfinite(norm) or norm > threshold:
            raise RuntimeError('Displacement PCG failed the original-equation residual audit')
        info = dict(kit=iterations+correction_iterations, norm_res=norm,
                    norm_res_log=[norm], coordinate_iterations=iterations,
                    correction_iterations=correction_iterations, linear_preconditioning=mode,
                    warm_start_requested=x0 is not None,
                    warm_start_accepted=initial is not None,
                    warm_start_projected=x0 is not None,
                    initial_norm_res_rel=(float(initial_norm/rhs_norm)
                                          if initial_norm is not None and rhs_norm > 0 else None))
        _finalize_cg_info(info, rhs_norm, threshold, par['maxiter'])
        info.update(true_norm_res_rel=info['norm_res_rel'], true_residual_converged=True)
        if not par.get('keep_solution_on_device', False):
            result = result.copy(val=cp.asnumpy(result.val))
        return result, info
    return solve
