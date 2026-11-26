"""Conjugate gradients for the strain formulation (NumPy or CuPy Tensors).

On CuPy, scalar products and the fused x/r update use the fixed-tree FP64
reductions of CupyCGReduction: results are deterministic and scalars stay on
the device except the residual norm checked once per iteration.
"""
import numpy as np
from cufhom_core.general.base import Timer
from cufhom_core.general.cupy_cg_reduction import CupyCGReduction
from cufhom_core.general.cupy_capacity import pinned_array,pinned_copy
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import (
    cp,
    cupy_synchronize,
    get_array_module,
    is_cupy_array,
    to_backend_array,
    to_host_array,
)


_CUPY_KERNELS = {}


def _cupy_kernels():
    """x/r update and p = r + beta p as single passes (scalar arguments on device)."""
    if not _CUPY_KERNELS:
        _CUPY_KERNELS['xr'] = cp.ElementwiseKernel(
            'T alpha, T p, T ap', 'T x, T r', 'x += alpha * p; r -= alpha * ap;', 'cufhom_cg_update_xr')
        _CUPY_KERNELS['p'] = cp.ElementwiseKernel(
            'T beta, T r', 'T p', 'p = r + beta * p;', 'cufhom_cg_update_p')
        _CUPY_KERNELS['x'] = cp.ElementwiseKernel(
            'T alpha, T p', 'T x', 'x += alpha * p;', 'cufhom_cg_host_x_update')
        _CUPY_KERNELS['r'] = cp.ElementwiseKernel(
            'T alpha, T ap', 'T r', 'r -= alpha * ap;', 'cufhom_cg_host_r_update')
        _CUPY_KERNELS['residual'] = cp.ElementwiseKernel(
            'T b, T ap', 'T r', 'r = b - ap;', 'cufhom_cg_host_initial_residual')
    return _CUPY_KERNELS


def _tensor_scalar_product_raw(y, x):
    assert isinstance(x, Tensor)
    assert y.val.shape == x.val.shape
    assert y.fft_form == x.fft_form

    xp = get_array_module(y.val, x.val)
    yval = y.val if xp is np else to_backend_array(y.val, prefer_backend='cupy')
    xval = x.val if xp is np else to_backend_array(x.val, prefer_backend='cupy')

    if y.Fourier:
        if x.fft_form in ['r']:
            if x.N[-1] % 2 == 1:
                scal = (
                    xp.sum(yval[..., 0] * xp.conj(xval[..., 0])).real +
                    2 * xp.sum(yval[..., 1:] * xp.conj(xval[..., 1:])).real
                ) / np.prod(y.N) ** 2
            else:
                scal = (
                    xp.sum(yval[..., 0] * xp.conj(xval[..., 0])).real +
                    xp.sum(yval[..., -1] * xp.conj(xval[..., -1])).real +
                    2 * xp.sum(yval[..., 1:-1] * xp.conj(xval[..., 1:-1])).real
                ) / np.prod(y.N) ** 2
        else:
            scal = xp.sum(yval[:] * xp.conj(xval[:])).real
    else:
        scal = xp.sum(yval[:] * xval[:]) / np.prod(y.N)
    return scal


def _sqrt_scalar_to_float(x):
    if is_cupy_array(x):
        return float(cp.sqrt(x).item())
    return np.double(x) ** 0.5


def _solver_tolerances(par):
    """Return relative/absolute tolerances while preserving the legacy alias."""
    rtol = float(par.get('rtol', par.get('tol', 1e-6)))
    atol = float(par.get('atol', 0.0))
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("CG rtol must be finite and non-negative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("CG atol must be finite and non-negative.")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("CG requires rtol > 0 or atol > 0.")
    return rtol, atol


def _finalize_cg_info(res, rhs_norm, threshold, maxiter):
    norm_res = float(res.get('norm_res', np.inf))
    denom = max(float(rhs_norm), np.finfo(float).tiny)
    res['rhs_norm'] = float(rhs_norm)
    res['threshold'] = float(threshold)
    res['norm_res_rel'] = float(norm_res / denom)
    res['converged'] = bool(np.isfinite(norm_res) and norm_res <= float(threshold))
    res['hit_maxiter'] = bool(res.get('kit', 0) >= int(maxiter) and not res['converged'])
    return res


def linear_solver(Afun, B, x0=None, par=None, solver='CG'):
    """CG for one Tensor right-hand side, or for stacked independent ones
    (par['batched_rhs']). Returns (x, info)."""
    if str(solver).lower() != 'cg':
        raise NotImplementedError("This kind (%s) of linear solver is not implemented" % solver)
    par = {} if par is None else par
    tim = Timer('Solving linsys by CG')
    if bool(par.get('batched_rhs', False)):
        x, info = CG_batched_tensor(Afun, B, x0=x0, par=par)
    else:
        x, info = CG(Afun, B, x0=x0, par=par)
    cupy_synchronize(x)
    tim.measure(print_time=False)
    if is_cupy_array(x.val) and not bool(par.get('keep_solution_on_device', False)):
        x = x.copy(val=to_host_array(x.val))
    info.update({'time': tim.vals})
    return x, info


def CG(Afun, B, x0=None, par=None):
    """Conjugate gradients for B = Afun(x) with Tensor fields.

    The residual is recursive; check_true_residual adds a diagnostic audit
    of B - Afun(x) at the end without changing the convergence test.
    """
    par = {} if par is None else par
    maxiter = int(par.get('maxiter', 1000))
    rtol, atol = _solver_tolerances(par)
    reduction = CupyCGReduction(B) if CupyCGReduction.supports(B, par) else None
    if (reduction is not None and B.val.dtype in (cp.float32,cp.float64)
            and getattr(B,'cufhom_resident_cg',False)):
        return _CG_resident_iterate(Afun,B,x0,par,reduction)
    if (reduction is not None and B.val.dtype in (cp.float32,cp.float64)
            and getattr(B,'cufhom_host_cg',False)):
        return _CG_host_iterate(Afun,B,x0,par,reduction)
    scal = reduction.dot if reduction is not None else (lambda X, Y: X*Y)

    if x0 is None:
        x = B.zeros_like()
        R = B.copy()
    else:
        x = x0
        R = B - Afun(x0)
    P = R.copy()
    rr = scal(R, R)
    rhs_norm = _sqrt_scalar_to_float(scal(B, B))
    threshold = max(atol, rtol * rhs_norm)
    res = {'kit': 0, 'norm_res': _sqrt_scalar_to_float(rr), 'restarts': 0, 'replacements': 0}
    norm_res_log = [res['norm_res']]
    while res['norm_res'] > threshold and res['kit'] < maxiter:
        res['kit'] += 1
        AP = Afun(P)
        pap = scal(P, AP)
        if reduction is None:
            from cufhom.fft.linear.common import LinearSolveError
            if not np.isfinite(float(pap)) or float(pap) <= 0 or not np.isfinite(float(rr)):
                raise LinearSolveError('Strain CG lost positive curvature', status='lost_curvature', iterations=res['kit'])
        alpha = rr / pap
        if reduction is None:
            alpha = float(alpha)
            if not np.isfinite(alpha) or alpha <= 0:
                raise LinearSolveError('Strain CG unsafe step', status='unsafe_step', iterations=res['kit'])
            rr_next = None
        else:
            rr_next = reduction.update_xr_rr(x, alpha, P, R, AP)
        if rr_next is None:
            if reduction is None:
                x += alpha*P
                R -= alpha*AP
            else:  # strided fields: same update without the fused reduction
                _cupy_kernels()['xr'](alpha, P.val, AP.val, x.val, R.val)
            rr_next = scal(R, R)
        del AP
        beta = rr_next / rr
        rr = rr_next
        res['norm_res'] = _sqrt_scalar_to_float(rr)
        norm_res_log.append(res['norm_res'])
        # A search direction is needed only for another iteration.
        if res['norm_res'] > threshold and res['kit'] < maxiter:
            if reduction is None:
                P *= beta
                P += R
            else:
                _cupy_kernels()['p'](beta, R.val, P.val)
    res['norm_res'] = _sqrt_scalar_to_float(rr)
    res['norm_res_log'] = [float(value) for value in norm_res_log]
    _finalize_cg_info(res, rhs_norm, threshold, maxiter)
    if bool(par.get('check_true_residual', False)):
        # Diagnostic audit: independently apply the operator to the final x.
        true_residual = B - Afun(x)
        true_norm = _sqrt_scalar_to_float(scal(true_residual, true_residual))
        res['true_norm_res'] = float(true_norm)
        res['true_norm_res_rel'] = float(true_norm / max(rhs_norm, np.finfo(float).tiny))
        res['true_residual_converged'] = bool(np.isfinite(true_norm) and true_norm <= threshold)
    return x, res


def _CG_resident_iterate(Afun,B,x0,par,reduction):
    """Keep x/r/p on GPU; save exact RHS bytes once and restore on every exit.

    The elasticity workflow validates the operator before opting in. Sharing
    its real/spectral scratch avoids a full extra field at the FFT boundary.
    The CG update and scalar-product tree are the ordinary resident ones.
    """
    rhs=pinned_copy(B.val)
    rhs_norm=_sqrt_scalar_to_float(reduction.dot(B,B))
    maxiter=int(par.get('maxiter',1000))
    rtol,atol=_solver_tolerances(par)
    threshold=max(atol,rtol*rhs_norm)
    x=B.zeros_like() if x0 is None else x0
    previous=getattr(Afun,'_capacity_inplace_fft',None)
    Afun._capacity_inplace_fft=True
    try:
        if x0 is not None:
            applied=Afun(x0)
            _cupy_kernels()['residual'](B.val,applied.val,B.val)
            del applied
        R=B.copy(val=B.val)
        P=R.copy()
        rr=reduction.dot(R,R)
        norm=_sqrt_scalar_to_float(rr)
        res=dict(kit=0,resident_cg_iterate=True,host_cg_iterate=False)
        history=[norm]
        while norm>threshold and res['kit']<maxiter:
            res['kit']+=1
            AP=Afun(P)
            alpha=rr/reduction.dot(P,AP)
            new=reduction.update_xr_rr(x,alpha,P,R,AP)
            if new is None:
                raise RuntimeError('Resident capacity CG requires contiguous FP32/FP64 fields')
            del AP
            beta=new/rr
            rr=new
            norm=_sqrt_scalar_to_float(rr)
            history.append(norm)
            if norm>threshold and res['kit']<maxiter:
                _cupy_kernels()['p'](beta,R.val,P.val)
        del R,P
    finally:
        B.val.set(rhs)
        cp.cuda.get_current_stream().synchronize()
        if previous is None:
            del Afun._capacity_inplace_fft
        else:
            Afun._capacity_inplace_fft=previous
    res.update(norm_res=norm,norm_res_log=history)
    _finalize_cg_info(res,rhs_norm,threshold,maxiter)
    if bool(par.get('check_true_residual',False)):
        true_residual=B-Afun(x)
        true_norm=_sqrt_scalar_to_float(reduction.dot(true_residual,true_residual))
        res.update(true_norm_res=true_norm,true_norm_res_rel=true_norm/max(rhs_norm,np.finfo(float).tiny),
                   true_residual_converged=bool(np.isfinite(true_norm) and true_norm<=threshold))
    return x,res


_HOST_CG_CHUNK = 16_777_216  # at most 64 MiB of FP32 device staging


def _CG_host_iterate(Afun,B,x0,par,reduction):
    """Same strain-CG recurrence, with a pinned iterate for large private loads.

    The known elasticity workflow marks B explicitly. Its device allocation
    doubles as the recursive residual; exact RHS bytes are retained on host
    and restored even after an exception. No caller-owned spectrum is used
    as scratch. P and AP remain GPU fields; x updates use FP32 GPU arithmetic
    on bounded chunks and the reduction tree is the resident CG tree.
    """
    rhs=pinned_copy(B.val)
    rhs_norm=_sqrt_scalar_to_float(reduction.dot(B,B))
    maxiter=int(par.get('maxiter',1000))
    rtol,atol=_solver_tolerances(par)
    threshold=max(atol,rtol*rhs_norm)
    host=pinned_array(B.val.shape,B.val.dtype)
    if x0 is None:
        host.fill(0)
        x=B.copy(val=host)
    else:
        x0.val.get(out=host)
        x=x0
    stage=cp.empty(min(B.val.size,_HOST_CG_CHUNK),dtype=B.val.dtype)
    kernels=_cupy_kernels()
    try:
        if x0 is not None:
            applied=Afun(x0)
            kernels['residual'](B.val,applied.val,B.val)
            del applied
            x.val=host
        R=B.copy(val=B.val)
        P=R.copy()
        rr=reduction.dot(R,R)
        norm=_sqrt_scalar_to_float(rr)
        res=dict(kit=0,norm_res=norm,host_cg_iterate=True)
        history=[norm]
        while norm>threshold and res['kit']<maxiter:
            res['kit']+=1
            AP=Afun(P)
            alpha=rr/reduction.dot(P,AP)
            a=cp.asarray(alpha,dtype=B.val.dtype)
            kernels['r'](a,AP.val,R.val)
            new=reduction.dot(R,R)
            flat=host.reshape(-1)
            direction=P.val.reshape(-1)
            for start in range(0,host.size,stage.size):
                width=min(stage.size,host.size-start)
                part=stage[:width]
                part.set(flat[start:start+width])
                kernels['x'](a,direction[start:start+width],part)
                part.get(out=flat[start:start+width])
            del AP
            beta=new/rr
            rr=new
            norm=_sqrt_scalar_to_float(rr)
            history.append(norm)
            if norm>threshold and res['kit']<maxiter:
                kernels['p'](beta,R.val,P.val)
        del P,R,stage
    finally:
        B.val.set(rhs)
        cp.cuda.get_current_stream().synchronize()
    x.val=cp.asarray(host)
    cp.cuda.get_current_stream().synchronize()
    res.update(norm_res=norm,norm_res_log=history)
    _finalize_cg_info(res,rhs_norm,threshold,maxiter)
    if bool(par.get('check_true_residual',False)):
        true_residual=B-Afun(x)
        true_norm=_sqrt_scalar_to_float(reduction.dot(true_residual,true_residual))
        res.update(true_norm_res=true_norm,true_norm_res_rel=true_norm/max(rhs_norm,np.finfo(float).tiny),
                   true_residual_converged=bool(np.isfinite(true_norm) and true_norm<=threshold))
    return x,res


_CUPY_BATCH_KERNELS = {}


def _batched_update_kernel(dtype, op):
    """Per-right-hand-side alpha/beta updates of Tensor(D, L, *N) fields."""
    key = (np.dtype(dtype).str, op)
    if key not in _CUPY_BATCH_KERNELS:
        ctype = 'float' if np.dtype(dtype) == np.float32 else 'double'
        if op == 'xr':
            code = r'''
            extern "C" __global__
            void cg_batch_update_xr(const T* __restrict__ alpha, const T* __restrict__ p,
                                    const T* __restrict__ ap, T* __restrict__ x, T* __restrict__ r,
                                    const long batch, const long spatial, const long total) {
                const long idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= total) return;
                const T a = alpha[(idx / spatial) % batch];
                x[idx] += a * p[idx];
                r[idx] -= a * ap[idx];
            }'''
        else:
            code = r'''
            extern "C" __global__
            void cg_batch_update_p(const T* __restrict__ beta, const T* __restrict__ r,
                                   T* __restrict__ p, const long batch, const long spatial,
                                   const long total) {
                const long idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= total) return;
                p[idx] = r[idx] + beta[(idx / spatial) % batch] * p[idx];
            }'''
        _CUPY_BATCH_KERNELS[key] = cp.RawKernel(code.replace('T', ctype), 'cg_batch_update_' + op)
    return _CUPY_BATCH_KERNELS[key]


def _batched_update(op, scalars, fields):
    """Launch a batched update on CuPy; False selects the array fallback.

    The kernel indexes flat memory: strided fields (e.g. the real part of a
    complex inverse FFT) must take the array fallback.
    """
    target = fields[-1].val
    if not (is_cupy_array(target) and target.ndim >= 3 and target.dtype in (cp.float32, cp.float64)
            and all(f.val.dtype == target.dtype and f.val.shape == target.shape
                    and f.val.flags.c_contiguous for f in fields)
            and is_cupy_array(scalars)):
        return False
    scalars = cp.ascontiguousarray(scalars.astype(target.dtype, copy=False))
    batch, spatial, total = int(target.shape[1]), int(np.prod(target.shape[2:])), int(target.size)
    _batched_update_kernel(target.dtype, op)(((total + 255) // 256,), (256,), (
        scalars, *(f.val for f in fields), np.int64(batch), np.int64(spatial), np.int64(total)))
    return True


def _batched_tensor_scalar_products(X, Y):
    xp = get_array_module(X.val, Y.val)
    xval = X.val if xp is np else to_backend_array(X.val, prefer_backend='cupy')
    yval = Y.val if xp is np else to_backend_array(Y.val, prefer_backend='cupy')

    axes = (0,) + tuple(range(2, xval.ndim))
    if X.Fourier:
        if X.fft_form in ['r']:
            raise NotImplementedError("batched_rhs con fft_form='r' aun no soporta scalar products en Fourier.")
        scal = xp.sum(xval * xp.conj(yval), axis=axes).real
    else:
        scal = xp.sum(xval * yval, axis=axes) / np.prod(X.N)
    return scal


def _batched_scale_view(vals, ndim):
    return vals.reshape((1, int(vals.shape[0])) + (1,) * (ndim - 2))


def CG_batched_tensor(Afun, B, x0=None, par=None):
    """
    Independent CG solves for several right-hand sides stored as Tensor(D, L, *N).

    This is not block-CG; each load has its own alpha/beta/residual, but the
    expensive Afun calls are batched so FFT backends can process multiple loads
    in one call.
    """
    par = {} if par is None else par
    maxiter = int(par.get('maxiter', 1000))
    rtol, atol = _solver_tolerances(par)
    if not isinstance(B, Tensor) or B.order != 2:
        raise ValueError("CG_batched_tensor requiere Tensor de orden 2 con shape=(D, n_rhs).")

    res = dict()
    if x0 is None:
        xCG = B.zeros_like()
        R = B.copy()
    else:
        xCG = x0
        R = B - Afun(x0)
    P = R.copy()
    rr = _batched_tensor_scalar_products(R, R)
    bb = _batched_tensor_scalar_products(B, B)
    xp = get_array_module(rr, bb)
    rhs_norm = xp.sqrt(bb)
    thresholds = xp.maximum(atol, rtol * rhs_norm)
    threshold_sq = thresholds * thresholds
    res['kit'] = 0
    threshold_host = to_host_array(thresholds)
    rhs_norm_host = to_host_array(rhs_norm)

    def check(rr):
        norm_res_host = to_host_array(xp.sqrt(rr))
        res['norm_res_per_rhs'] = norm_res_host.astype(float).tolist()
        res['norm_res'] = float(np.max(norm_res_host)) if norm_res_host.size else 0.0
        res['converged'] = bool(np.all(norm_res_host <= threshold_host))
    check(rr)

    while (not res['converged']) and (res['kit'] < maxiter):
        res['kit'] += 1
        AP = Afun(P)
        denom = _batched_tensor_scalar_products(P, AP)
        active = rr > threshold_sq
        safe_denom = xp.where(active, denom, 1.0)
        alp = xp.where(active, rr / safe_denom, 0.0)
        if not _batched_update('xr', alp, (P, AP, xCG, R)):
            alpha_view = _batched_scale_view(alp, xCG.val.ndim)
            xCG.val += alpha_view * P.val
            R.val -= alpha_view * AP.val

        rrnext = _batched_tensor_scalar_products(R, R)
        safe_rr = xp.where(active, rr, 1.0)
        next_active = rrnext > threshold_sq
        bet = xp.where(active & next_active, rrnext / safe_rr, 0.0)
        rr = rrnext
        check(rr)

        if not res['converged'] and res['kit'] < maxiter:
            if not _batched_update('p', bet, (R, P)):
                beta_view = _batched_scale_view(bet, P.val.ndim)
                P.val *= beta_view
                P.val += R.val

    norm_res_host = to_host_array(xp.sqrt(rr))
    denom_host = np.maximum(rhs_norm_host, np.finfo(float).tiny)
    converged_per_rhs = norm_res_host <= threshold_host
    res['norm_res_per_rhs'] = norm_res_host.astype(float).tolist()
    res['norm_res_rel_per_rhs'] = (norm_res_host / denom_host).astype(float).tolist()
    res['rhs_norm_per_rhs'] = rhs_norm_host.astype(float).tolist()
    res['threshold_per_rhs'] = threshold_host.astype(float).tolist()
    res['converged_per_rhs'] = converged_per_rhs.astype(bool).tolist()
    res['norm_res'] = float(np.max(norm_res_host)) if norm_res_host.size else 0.0
    res['norm_res_rel'] = float(np.max(norm_res_host / denom_host)) if norm_res_host.size else 0.0
    res['converged'] = bool(np.all(converged_per_rhs))
    res['hit_maxiter'] = bool(res['kit'] >= maxiter and not res['converged'])
    return xCG, res
