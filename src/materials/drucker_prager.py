"""Pressure-dependent Drucker-Prager plasticity, Mandel/GPa, tension positive.

f = q + friction*mean_stress - (yield_stress + hardening*k).
g = q + beta*mean_stress, beta = dilatancy (friction when omitted).
k accumulates the plastic multiplier, including at the nonsmooth cone apex.
Backward Euler has closed-form smooth/apex returns and an exact, generally
nonsymmetric compact tangent. No 6x6 tangent field is stored.
"""

from dataclasses import dataclass

import numpy as np
from cufhom.materials.plasticity import (
    J2Material,
    MaterialResponse,
    PlasticState,
    elastic_action,
)


@dataclass(frozen=True)
class DruckerPragerMaterial(J2Material):
    """Configure linear-hardening Drucker-Prager plasticity and dilatancy."""

    friction: float = 0.3
    dilatancy: float | None = None

    def __post_init__(self):
        """Validate the inherited material and Drucker-Prager parameters."""
        super().__post_init__()
        if self.hardening <= 0:
            raise ValueError("Drucker-Prager requires hardening > 0")
        if self.nonlinear_hardening or self.kinematic_hardening or self.viscosity:
            raise ValueError("Drucker-Prager supports linear isotropic hardening only")
        if (
            not np.isfinite([self.friction, self.beta]).all()
            or not 0 <= self.beta <= self.friction < 3
        ):
            raise ValueError("Require finite 0 <= dilatancy <= friction < 3")

    @property
    def beta(self):
        """Return the dilatancy coefficient, defaulting to associated flow."""
        return self.friction if self.dilatancy is None else self.dilatancy

    @property
    def model_name(self):
        """Return the constitutive model identifier used in saved configuration."""
        return "small_strain_Drucker_Prager_linear_hardening"

    @property
    def associated(self):
        """Return whether the plastic flow and yield surface use the same coefficient."""
        return self.beta == self.friction

    @classmethod
    def from_strengths(cls, tensile, compressive, **kwargs):
        """Fit the initial cone to positive uniaxial strengths, in GPa.

        Dilatancy and hardening require independent data; no angle conversion
        or calibration of those parameters is inferred from these strengths.
        """
        if (
            not np.isfinite([tensile, compressive]).all()
            or not 0 < tensile <= compressive
        ):
            raise ValueError("Require 0 < tensile strength <= compressive strength")
        if "friction" in kwargs or "yield_stress" in kwargs:
            raise ValueError("Strengths determine friction and yield_stress")
        return cls(
            friction=3 * (compressive - tensile) / (compressive + tensile),
            yield_stress=2 * compressive * tensile / (compressive + tensile),
            **kwargs,
        )


def update_cpu(strain, state, material, ids, table, start=None):
    """Array reference for the closed-form backward Euler map, including apex."""
    if start is not None:
        raise ValueError(
            "Drucker-Prager uses backward_euler; midpoint is not implemented"
        )
    mu, K, alpha, beta, H = (
        material.mu,
        material.K,
        material.friction,
        material.beta,
        material.hardening,
    )
    ee = strain - state.plastic_strain
    mean = K * np.sum(ee[:3], axis=0)
    trial = 2 * mu * ee
    trial[:3] -= (2 * mu / 3) * np.sum(ee[:3], axis=0)
    norm = np.sqrt(np.sum(trial * trial, axis=0))
    q = np.sqrt(1.5) * norm
    u = trial / np.maximum(norm, np.finfo(strain.dtype).tiny)
    radius = material.yield_stress + H * state.equivalent
    apex_denom = alpha * K * beta + H
    denom = 3 * mu + apex_denom
    dp = np.where(ids == 0, np.maximum(q + alpha * mean - radius, 0) / denom, 0)
    # q - 3*mu*dp <= 0 iff alpha*mean-radius >= q*apex_denom/(3*mu).
    # The latter does not spuriously select the apex through cancellation
    # when friction vanishes and the strictly positive radius is tiny.
    apex = (dp > 0) & (alpha * mean - radius >= q * (apex_denom / (3 * mu)))
    dp = np.where(apex, (alpha * mean - radius) / apex_denom, dp)
    plastic = dp > 0
    smooth_q = (apex_denom / denom) * q + (3 * mu / denom) * (radius - alpha * mean)
    ratio = np.ones_like(q)
    np.divide(smooth_q, q, out=ratio, where=plastic & ~apex)
    ratio = np.where(apex, 0, np.maximum(ratio, 0))
    stress = trial * ratio
    # At the apex alpha > 0 and alpha*mean_new = radius + H*dp.
    # Do not subtract two large pressures near incompressibility.
    returned_mean = np.array(mean - K * beta * dp, copy=True)
    np.divide(radius + H * dp, alpha, out=returned_mean, where=apex)
    stress[:3] += returned_mean
    a = 2 * mu * ratio
    b = np.where(plastic, 2 * mu * (apex_denom / denom - ratio), 0)
    bulk = np.where(plastic, K * ((3 * mu + H) / denom), K)
    d = np.where(plastic, -2 * mu * np.sqrt(1.5) * K * alpha / denom, 0)
    c = np.where(plastic, -2 * mu * np.sqrt(1.5) * K * beta / denom, 0)
    a = np.where(apex, 0, a)
    b = np.where(apex, 0, b)
    bulk = np.where(apex, K * H / apex_denom, bulk)
    d = np.where(apex, 0, d)
    c = np.where(apex, 0, c)
    direction = u.copy()
    direction = np.where(
        apex[None],
        trial / np.maximum(2 * mu * np.sqrt(1.5) * dp, np.finfo(strain.dtype).tiny),
        direction,
    )
    direction[:3] += beta / (3 * np.sqrt(1.5))
    direction = np.where(plastic[None], direction, 0)
    if np.any(ids):
        fiber = elastic_action(strain, ids, table)
        stress[:, ids != 0] = fiber[:, ids != 0]
    return MaterialResponse(stress, direction, a, b, dp, bulk, d, c)


def tangent_cpu(value, response, material, ids, table):
    """Apply the pressure-dependent algorithmic tangent on host arrays."""
    trace = np.sum(value[:3], axis=0)
    # Commit direction includes the volumetric part; tangent uses unit
    # deviatoric trial direction, reconstructed by removing that constant.
    u = response.direction.copy()
    u[:3] -= material.beta / (3 * np.sqrt(1.5))
    length = np.sqrt(np.sum(u * u, axis=0))
    u /= np.maximum(length, np.finfo(value.dtype).tiny)
    dot = np.sum(u * value, axis=0)
    out = (
        response.a * value
        + response.b * u * dot
        + response.coupling_to_direction * u * trace
    )
    out[:3] += (
        response.bulk - response.a / 3
    ) * trace + response.coupling_to_pressure * dot
    if np.any(ids):
        fiber = elastic_action(value, ids, table)
        out[:, ids != 0] = fiber[:, ids != 0]
    return out


_CUDA = r"""
extern "C" __global__ void dp_update(const REAL* e,const REAL* ep,const REAL* k,
 const int* ids,const REAL* table,REAL* sig,REAL* u,REAL* aa,REAL* bb,REAL* dp,
 REAL* bulk,REAL* cd,REAL* cp,long long n,REAL mu,REAL K,REAL sy,REAL H,REAL alpha,REAL beta) {
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){
   for(int i=0;i<6;i++){REAL v=0;for(int j=0;j<6;j++)v+=table[ids[z]*36+i*6+j]*e[j*n+z];sig[i*n+z]=v;u[i*n+z]=0;}
   aa[z]=2*mu;bb[z]=0;dp[z]=0;bulk[z]=K;cd[z]=0;cp[z]=0;return;
 }
 REAL tr=0,s[6],n2=0;const REAL r15=sqrt((REAL)1.5);
 for(int i=0;i<3;i++)tr+=e[i*n+z]-ep[i*n+z];
 const REAL mt=K*tr;
 for(int i=0;i<6;i++){s[i]=2*mu*(e[i*n+z]-ep[i*n+z]-(i<3?tr/3:0));n2+=s[i]*s[i];}
 const REAL sn=sqrt(n2),qt=r15*sn,Y=sy+H*k[z],Da=alpha*K*beta+H,D=3*mu+Da;
 REAL gamma=fmax((REAL)0,(qt+alpha*mt-Y)/D);
 bool apex=gamma>0&&alpha*mt-Y>=qt*(Da/(3*mu));
 if(apex)gamma=(alpha*mt-Y)/Da;
 const bool plastic=gamma>0;
 const REAL t=apex?0:(plastic?fmax((REAL)0,((Da/D)*qt+(3*mu/D)*(Y-alpha*mt))/qt):(REAL)1);
 const REAL mean=apex?(Y+H*gamma)/alpha:mt-K*beta*gamma;
 aa[z]=2*mu*t;bb[z]=apex?0:(plastic?2*mu*(Da/D-t):0);
 bulk[z]=apex?K*H/Da:(plastic?K*((3*mu+H)/D):K);
 cd[z]=(!plastic||apex)?0:-2*mu*r15*K*alpha/D;
 cp[z]=(!plastic||apex)?0:-2*mu*r15*K*beta/D;dp[z]=gamma;
 for(int i=0;i<6;i++){
   // dp direction is the complete flow increment / (sqrt(1.5)*gamma),
   // including the apex subgradient and plastic volume. Existing commit
   // then allocates a new state without touching the confirmed one.
   u[i*n+z]=plastic?((apex?s[i]/(2*mu*r15*gamma):s[i]/sn)+(i<3?beta/(3*r15):0)):0;
   REAL v=t*s[i]+(i<3?mean:0);
   sig[i*n+z]=v;
 }
}
extern "C" __global__ void dp_tangent(const REAL* v,const REAL* u,const REAL* a,const REAL* b,
 const REAL* bulk,const REAL* cd,const REAL* cp,const int* ids,const REAL* table,
 REAL* out,long long n,REAL beta){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){for(int i=0;i<6;i++){REAL w=0;for(int j=0;j<6;j++)w+=table[ids[z]*36+i*6+j]*v[j*n+z];out[i*n+z]=w;}return;}
 REAL tr=0,d=0,n2=0,uv[6];const REAL vol=beta/(3*sqrt((REAL)1.5));
 for(int i=0;i<6;i++){uv[i]=u[i*n+z]-(i<3?vol:0);n2+=uv[i]*uv[i];}
 const REAL inv=n2>0?1/sqrt(n2):0;
 for(int i=0;i<6;i++){uv[i]*=inv;d+=uv[i]*v[i*n+z];if(i<3)tr+=v[i*n+z];}
 for(int i=0;i<6;i++){
   REAL w=a[z]*v[i*n+z]+b[z]*uv[i]*d+cd[z]*uv[i]*tr+(i<3?(bulk[z]-a[z]/3)*tr+cp[z]*d:0);
   out[i*n+z]=w;
 }
}
extern "C" __global__ void dp_audit(const REAL* sig,const REAL* dp,const REAL* k,const int* ids,
 REAL* violation,REAL* work,long long n,REAL sy,REAL H,REAL alpha,REAL beta){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 const REAL mean=(sig[z]+sig[n+z]+sig[2*n+z])/3;REAL q2=0;
 for(int i=0;i<6;i++){const REAL s=sig[i*n+z]-(i<3?mean:0);q2+=s*s;}
 const REAL q=sqrt((REAL)1.5*q2),Y=sy+H*(k[z]+dp[z]);
 const REAL f=q+alpha*mean-Y;
 violation[z]=ids[z]?0:(isnan(f)?f:fmax(f,(REAL)0));
 // Irreversible work after subtracting isotropic hardening energy.
 work[z]=ids[z]?0:dp[z]*(sy+(beta-alpha)*mean);
}
extern "C" __global__ void dp_commit(const REAL* ep,const REAL* k,const REAL* u,const REAL* dp,
 REAL* ep_out,REAL* k_out,long long n,REAL r15){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 for(int i=0;i<6;i++){
   const long long j=i*n+z;
   const REAL scaled=MUL_RN(r15,u[j]);const REAL delta=MUL_RN(scaled,dp[z]);
   ep_out[j]=ADD_RN(ep[j],delta);
 }
 k_out[z]=ADD_RN(k[z],dp[z]);
}
"""


class CuPyDruckerPrager:
    """One CUDA thread per voxel; compact tangent, elastic fiber branch."""

    def __init__(self, cp, material, ids, table, dtype):
        self.cp, self.material, self.ids, self.table = cp, material, ids, table
        self.dtype = np.dtype(dtype).type
        self.shape, self.n = ids.shape, ids.size
        kind = "f" if self.dtype == np.float32 else "d"
        code = _CUDA.replace("REAL", "float" if kind == "f" else "double")
        code = code.replace("MUL_RN", f"__{kind}mul_rn").replace(
            "ADD_RN", f"__{kind}add_rn"
        )
        mod = cp.RawModule(code=code)
        self.update_kernel = mod.get_function("dp_update")
        self.tangent_kernel = mod.get_function("dp_tangent")
        self.audit_kernel = mod.get_function("dp_audit")
        self.commit_kernel = mod.get_function("dp_commit")

    def update(self, strain, state, start=None):
        """Evaluate the device backward-Euler return map without committing state."""
        if start is not None:
            raise ValueError(
                "Drucker-Prager uses backward_euler; midpoint is not implemented"
            )
        cp, m, d = self.cp, self.material, self.dtype
        fields = [cp.empty(self.shape, dtype=d) for _ in range(6)]
        r = MaterialResponse(
            cp.empty_like(strain), cp.empty_like(strain), *fields[:3], *fields[3:]
        )
        self.update_kernel(
            ((self.n + 255) // 256,),
            (256,),
            (
                strain,
                state.plastic_strain,
                state.equivalent,
                self.ids,
                self.table,
                r.stress,
                r.direction,
                r.a,
                r.b,
                r.dp,
                r.bulk,
                r.coupling_to_direction,
                r.coupling_to_pressure,
                np.int64(self.n),
                d(m.mu),
                d(m.K),
                d(m.yield_stress),
                d(m.hardening),
                d(m.friction),
                d(m.beta),
            ),
        )
        return r

    def tangent(self, value, response):
        """Apply the pressure-dependent device tangent to a strain increment."""
        out = self.cp.empty_like(value)
        self.tangent_kernel(
            ((self.n + 255) // 256,),
            (256,),
            (
                value,
                response.direction,
                response.a,
                response.b,
                response.bulk,
                response.coupling_to_direction,
                response.coupling_to_pressure,
                self.ids,
                self.table,
                out,
                np.int64(self.n),
                self.dtype(self.material.beta),
            ),
        )
        return out

    def audit(self, response, state):
        """Return device yield-violation and plastic-work fields for the trial response."""
        violation = self.cp.empty(self.shape, dtype=self.dtype)
        work = self.cp.empty_like(violation)
        m, d = self.material, self.dtype
        self.audit_kernel(
            ((self.n + 255) // 256,),
            (256,),
            (
                response.stress,
                response.dp,
                state.equivalent,
                self.ids,
                violation,
                work,
                np.int64(self.n),
                d(m.yield_stress),
                d(m.hardening),
                d(m.friction),
                d(m.beta),
            ),
        )
        return violation, work

    def commit(self, state, response):
        """One pass, fresh outputs, original separate-operation rounding."""
        ep = self.cp.empty_like(state.plastic_strain)
        k = self.cp.empty_like(state.equivalent)
        self.commit_kernel(
            ((self.n + 255) // 256,),
            (256,),
            (
                state.plastic_strain,
                state.equivalent,
                response.direction,
                response.dp,
                ep,
                k,
                np.int64(self.n),
                self.dtype(np.sqrt(1.5)),
            ),
        )
        return PlasticState(ep, k)
