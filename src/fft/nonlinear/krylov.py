"""Newton tangent solvers in displacement coordinates (float32 GPU).

TangentPipeline runs CG/GMRES on the fused axis-split FFT pipeline of the
linear solver, with the compact J2/Drucker-Prager tangent as material stage.
FusedCoordinatePCG keeps the production FFTs; it serves PTC shifts and grids
the pipeline does not support. Its isotropic reference is evaluated while
updating the direction, and r.r / r.M^-1.r are reduced together during the
x/r update, preserving the physical/isometric residual norm.
"""

import operator
import os
from types import SimpleNamespace

import numpy as np
from cufhom.fft.linear.energy import energy_update_device
from cufhom.fft.linear.fused import FusedPipeline, FusedStep, grid_supported
from cufhom.fft.linear.krylov import SymmetricCoordinates, fused_cg
from cufhom_core.tensors.fft import cp

_MODULE = None


def _solve_controls(tolerance, maxiter):
    tolerance = float(tolerance)
    maxiter = operator.index(maxiter)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Krylov tolerance must be finite and between zero and one")
    if maxiter < 1:
        raise ValueError("Krylov maxiter must be positive")
    return tolerance, maxiter


_CODE = r"""
__device__ __forceinline__ void reference3(const float2* r, float2* z,
 long long s,long long n0,long long n1,long long nz,float iy0,float iy1,float iy2,
 float a,float b,long long n2) {
 if(s==0){for(int c=0;c<3;c++)z[c]=make_float2(0,0);return;}
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 // Even-grid Nyquist: zero wavenumber; a vanishing wavevector is excluded.
 float k[3]={(n0%2==0&&i0==n0/2)?0:(i0<=n0/2?i0:i0-n0)*iy0,
             (n1%2==0&&i1==n1/2)?0:(i1<=n1/2?i1:i1-n1)*iy1,(n2%2==0&&i2==n2/2)?0:i2*iy2};
 const float kk=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];
 if(kk==0){for(int c=0;c<3;c++)z[c]=make_float2(0,0);return;}
 float inv=1/kk,re=0,im=0;
 for(int c=0;c<3;c++){re+=k[c]*r[c].x;im+=k[c]*r[c].y;}
 for(int c=0;c<3;c++)z[c]=make_float2(a*r[c].x+(b-a)*k[c]*re*inv,
                                   a*r[c].y+(b-a)*k[c]*im*inv);
}
__device__ __forceinline__ void finish2(double rr,double rz,double* out) {
 for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);rz+=__shfl_down_sync(0xffffffff,rz,s);}
 __shared__ double rr_w[8],rz_w[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0){rr_w[warp]=rr;rz_w[warp]=rz;}
 __syncthreads();
 if(warp==0){
   rr=lane<8?rr_w[lane]:0;rz=lane<8?rz_w[lane]:0;
   for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);rz+=__shfl_down_sync(0xffffffff,rz,s);}
   if(lane==0){out[blockIdx.x]=rr;out[gridDim.x+blockIdx.x]=rz;}
 }
}
extern "C" __global__ void update_energy(const float2* p,const float2* ap,
 float alpha,float2* x,float2* r,double* partial,long long n0,long long n1,long long nz,
 float iy0,float iy1,float iy2,float a,float b,const float* mean,int free_mask,long long n2) {
 long long count=n0*n1*nz;
 double rr=0,rz=0;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;
     s+=(long long)gridDim.x*blockDim.x){
   float2 rc[3],z[3];
   for(int c=0;c<3;c++){
     long long j=c*count+s;
     float2 xi=x[j],ri=r[j],pi=p[j],api=ap[j];
     xi.x+=alpha*pi.x;xi.y+=alpha*pi.y;
     ri.x-=alpha*api.x;ri.y-=alpha*api.y;
     x[j]=xi;r[j]=rc[c]=ri;
   }
   reference3(rc,z,s,n0,n1,nz,iy0,iy1,iy2,a,b,n2);
   for(int c=0;c<3;c++){
     rr+=(double)rc[c].x*rc[c].x+(double)rc[c].y*rc[c].y;
     rz+=(double)rc[c].x*z[c].x+(double)rc[c].y*z[c].y;
   }
   if(s==0){
     float rm[6];int i=0;
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       long long j=3*count+i++;
       float2 xi=x[j],ri=r[j],pi=p[j],api=ap[j];
       xi.x+=alpha*pi.x;xi.y+=alpha*pi.y;
       ri.x-=alpha*api.x;ri.y-=alpha*api.y;
       x[j]=xi;r[j]=ri;rm[c]=ri.x;
       rr+=(double)ri.x*ri.x+(double)ri.y*ri.y;
     }
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       float v=0;
       for(int d=0;d<6;d++)if(free_mask&(1<<d))v+=mean[6*c+d]*rm[d];
       rz+=(double)rm[c]*v;
     }
   }
 }
 finish2(rr,rz,partial);
}
extern "C" __global__ void direction(const float2* r,float beta,float2* p,
 long long n0,long long n1,long long nz,float iy0,float iy1,float iy2,
 float a,float b,const float* mean,int free_mask,long long n2) {
 long long count=n0*n1*nz;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;
     s+=(long long)gridDim.x*blockDim.x){
   float2 rc[3],z[3];
   for(int c=0;c<3;c++)rc[c]=r[c*count+s];
   reference3(rc,z,s,n0,n1,nz,iy0,iy1,iy2,a,b,n2);
   for(int c=0;c<3;c++){
     long long j=c*count+s;float2 v=p[j];
     p[j]=make_float2(z[c].x+beta*v.x,z[c].y+beta*v.y);
   }
   if(s==0){
     int i=0;
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       float v=0;int j=0;
       for(int d=0;d<6;d++)if(free_mask&(1<<d))v+=mean[6*c+d]*r[3*count+j++].x;
       long long index=3*count+i++;float2 old=p[index];
       p[index]=make_float2(v+beta*old.x,beta*old.y);
     }
   }
 }
}
"""


class FusedCoordinatePCG:
    """Fixed reference per solve; supports the free mean modes and PTC shift."""

    def __init__(self, coordinates):
        global _MODULE
        if cp is None or coordinates.xp is not cp:
            raise ValueError("Fused coordinate PCG requires CuPy")
        self.Q = coordinates
        self.blocks = min(16384, (coordinates.count + 255) // 256)
        self.partial = cp.empty((2, self.blocks), dtype=cp.float64)
        self.geometry = (
            *(np.int64(n) for n in coordinates.hshape),
            *(np.float32(1 / L) for L in coordinates.P.lengths),
        )
        self.mask = np.int32(sum(1 << i for i in coordinates.P.free))
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_CODE)
        self.update_kernel = _MODULE.get_function("update_energy")
        self.direction_kernel = _MODULE.get_function("direction")

    def _reference_args(self, reference):
        if reference.P is not self.Q.P or reference.mean.dtype != cp.float32:
            raise ValueError("Reference must share the projector and use float32")
        return (
            np.float32(reference.a),
            np.float32(reference.b),
            reference.mean,
            self.mask,
            np.int64(self.Q.shape[-1]),
        )

    def update(self, p, ap, alpha, x, r, reference):
        """Update the solution and residual and apply reference preconditioning."""
        self.update_kernel(
            (self.blocks,),
            (256,),
            (
                p,
                ap,
                np.float32(alpha),
                x,
                r,
                self.partial,
                *self.geometry,
                *self._reference_args(reference),
            ),
        )
        return cp.asnumpy(cp.sum(self.partial, axis=1, dtype=cp.float64))

    def direction(self, r, beta, p, reference):
        """Update the Krylov direction using the preconditioned residual."""
        self.direction_kernel(
            (self.blocks,),
            (256,),
            (r, np.float32(beta), p, *self.geometry, *self._reference_args(reference)),
        )

    def solve(self, action, rhs, dot, tolerance, maxiter, reference):
        """Solve a compatible FP32 linear system with high-precision scalar reductions."""
        from cufhom.fft.krylov import NonlinearConvergenceError

        tolerance, maxiter = _solve_controls(tolerance, maxiter)
        if (
            rhs.dtype != cp.complex64
            or rhs.ndim != 1
            or rhs.size != self.Q.size
            or not rhs.flags.c_contiguous
        ):
            raise ValueError(
                "Expected contiguous complex64 coordinates on the configured grid"
            )
        x, r = cp.zeros_like(rhs), rhs.copy()
        rr = dot(r, r)
        if not np.isfinite(rr) or rr < 0:
            raise NonlinearConvergenceError("Nonfinite or negative PCG RHS norm")
        if rr == 0:
            from cufhom.fft.nonlinear.gmres import _nonzero_storage

            if _nonzero_storage(rhs):
                raise NonlinearConvergenceError("PCG RHS norm underflowed")
            return x, 0
        limit = tolerance * tolerance * rr
        p = self.Q.precondition(r, reference)
        rz = dot(r, p)
        for iteration in range(maxiter):
            ap = action(p)
            pap = dot(p, ap)
            if not np.isfinite([pap, rz]).all() or min(pap, rz) <= 0:
                raise NonlinearConvergenceError("Fused coordinate PCG lost positivity")
            # The device update uses FP32 even though the dots are FP64.
            # Reject an unrepresentable step before it corrupts x and r.
            with np.errstate(over="ignore", under="ignore"):
                alpha = np.float32(rz / pap)
            if not np.isfinite(alpha) or alpha <= 0:
                raise NonlinearConvergenceError("Unsafe fused coordinate PCG step")
            rr, new_rz = self.update(p, ap, alpha, x, r, reference)
            del ap
            if not np.isfinite([rr, new_rz]).all() or rr < 0:
                raise NonlinearConvergenceError(
                    "Nonfinite or negative fused coordinate residual"
                )
            if rr <= limit or iteration + 1 == maxiter:
                # These fields are replaced after an audit. Do not overlap
                # their storage with the production FFT's strain/stress work.
                del p, r
                r = rhs - action(x)
                rr = dot(r, r)
                if not np.isfinite(rr) or rr < 0:
                    raise NonlinearConvergenceError(
                        "Nonfinite or negative true coordinate residual"
                    )
                if rr <= limit:
                    return x, iteration + 1
                if iteration + 1 == maxiter:
                    break
                p = self.Q.precondition(r, reference)
                new_rz = dot(r, p)
                if not np.isfinite(new_rz) or new_rz <= 0:
                    raise NonlinearConvergenceError(
                        "Fused coordinate PCG lost positivity"
                    )
            else:
                if new_rz <= 0:
                    raise NonlinearConvergenceError(
                        "Fused coordinate PCG lost positivity"
                    )
                with np.errstate(over="ignore", under="ignore"):
                    beta = np.float32(new_rz / rz)
                if not np.isfinite(beta):
                    raise NonlinearConvergenceError(
                        "Unsafe fused coordinate PCG direction"
                    )
                self.direction(r, beta, p, reference)
            rz = new_rz
        raise NonlinearConvergenceError(
            f"Fused coordinate PCG failed true residual after {maxiter} iterations"
        )


class TangentPipeline:
    """(P C_alg P) de = -P sigma on the fused pipeline, in whitened coordinates.

    The operator is R Q* C_alg Q R, with R = M^1/2 of the same isotropic
    reference as FusedCoordinatePCG (free mean strains included), so CG and
    PCG produce the same iterates; only the stopping norm becomes the
    preconditioned one. Each iteration is five kernels over one hybrid
    spectrum (CUDA graphs on small grids); the true residual is recomputed
    through the pipeline before a correction is accepted. No residual
    smoothing: it minimizes |r|, which weights the soft plastic modes by
    lambda^2, and the accepted states lost about half a digit of accuracy.
    """

    def __init__(self, projector, material, ids, table):
        self.Q = SymmetricCoordinates(projector, material.K, material.mu)
        self.fallbacks = 0  # BiCGStab solves completed by GMRES
        rows, cols = np.triu_indices(6)
        sym21 = cp.ascontiguousarray(table[:, rows, cols], dtype=cp.float32)
        self.pipeline = FusedPipeline(
            self.Q,
            SimpleNamespace(val=ids, material_table=sym21),
            j2_bulk=material.K,
            pressure_beta=getattr(material, "beta", None),
        )

    @staticmethod
    def supported(shape):
        """Check whether the shape supports the fused nonlinear FFT route."""
        return grid_supported(shape)

    def solve(
        self,
        response,
        residual,
        tolerance,
        maxiter,
        dot,
        method="cg",
        restart=12,
        shift=0.0,
    ):
        """Return the correction strain Q R z and Krylov iteration count.

        residual holds the isometric displacement coordinates Q*(P sigma);
        the whitened right-hand side R Q* sigma needs no transform.
        """
        tolerance, maxiter = _solve_controls(tolerance, maxiter)
        if method not in ("cg", "gmres", "bicgstab"):
            raise ValueError("Tangent method must be 'cg', 'bicgstab' or 'gmres'")
        if method != "cg":
            restart = operator.index(restart)
            if restart < 1:
                raise ValueError("GMRES restart must be positive")
        pipeline = self.pipeline
        try:
            pipeline.bind(
                response.a,
                response.b,
                response.direction,
                response.bulk,
                response.coupling_to_direction,
                response.coupling_to_pressure,
            )
            # A mixed-precision Newton step whitens its FP64 residual and
            # solves in the pipeline's precision.
            rhs = pipeline.to_fused(
                self.Q.whiten(residual).astype(pipeline.complex_dtype, copy=False)
            )
            cp.negative(rhs, out=rhs)
            if method == "bicgstab":
                from cufhom.fft.nonlinear.gmres import nonsymmetric_solve

                if shift:

                    def into(z, out=None):
                        out = pipeline.matvec(z, out)
                        out += out.dtype.type(shift) * z
                        return out
                else:
                    into = pipeline.matvec
                z, count, fallback = nonsymmetric_solve(
                    into,
                    rhs,
                    dot,
                    tolerance,
                    maxiter,
                    restart=restart,
                    action_into=into,
                )
                self.fallbacks += fallback
            elif method == "gmres":
                from cufhom.fft.nonlinear.gmres import gmres

                z, count = gmres(
                    pipeline.matvec, rhs, dot, tolerance, maxiter, restart=restart
                )
            elif (
                pipeline.config["graph_iterations"] > 0
                and os.environ.get("CUFHOM_CUDA_GRAPHS", "1") != "0"
            ):
                z, count = pipeline.graph_cg(
                    rhs, dot, tolerance, maxiter, pipeline.matvec, smoothing=False
                )
            else:
                step = FusedStep(pipeline)
                pipeline.pending = None
                z, count = fused_cg(
                    pipeline.matvec,
                    rhs,
                    dot,
                    tolerance,
                    maxiter,
                    action_and_dot=lambda p, direction=None: (
                        step,
                        pipeline.apply(p, direction),
                    ),
                    device_update_callback=energy_update_device,
                    settle=pipeline.settle,
                )
        finally:
            # Failed host CG may leave a deferred update holding its vectors.
            # They will not be consumed; retain the pipeline's spectrum/graph
            # allocation policy through release(), including on exceptions.
            pipeline.pending = None
            pipeline.release()
        return self.Q.strain(pipeline.to_production(z)), count
