"""Paraboloidal pressure-dependent plasticity of polymer matrices, Mandel/GPa.

Tschoegl's yield paraboloid with the non-associated potential of Melro et al.
(2013), tension positive, mean stress m = tr(sigma)/3, von Mises stress q:

  f = q^2 - 3 (sigma_t - sigma_c) m - sigma_t sigma_c,
  g = q^2 + alpha m^2,  alpha = 9/2 (1 - 2 nu_p)/(1 + nu_p),
  dep = gamma dg/dsigma = gamma (3 s + 2/3 alpha m I),
  dp = sqrt(dep:dep/(1 + 2 nu_p^2))  (axial plastic strain in uniaxial tests).

sigma_t(p) and sigma_c(p) are piecewise-linear hardening curves of the
equivalent plastic strain p, constant beyond their last knot. f reproduces
both uniaxial yield stresses, and nu_p is the plastic Poisson ratio of
uniaxial tests. Backward Euler gives the closed forms s = s_trial/(1+6 mu
gamma), m = m_trial/(1+2 K alpha gamma); the multiplier is the root of the
scalar f(gamma) = 0 (safeguarded Newton with a bracket). The consistent
tangent is exact and nonsymmetric, with the compact form of Drucker-Prager:

  a v + b u(u.v) + c_d u tr(v) + [(bulk - a/3) tr(v) + c_p (u.v)] I,

u the unit trial deviator, stored as the response direction. The plastic
strain increment is rebuilt at commit from the accepted stress and dp.
No 6x6 tangent field is stored.
"""

from dataclasses import asdict, dataclass

import numpy as np
from cufhom.materials.plasticity import MaterialResponse, PlasticState, elastic_action


def _curve(knots, name):
    knots = np.asarray(knots, dtype=np.float64)
    if (
        knots.ndim != 2
        or knots.shape[1] != 2
        or not 1 <= len(knots) <= 32
        or not np.isfinite(knots).all()
    ):
        raise ValueError(
            f"{name} must hold 1 to 32 finite [plastic_strain, stress] knots"
        )
    p, stress = knots.T
    if (
        p[0] != 0
        or np.any(np.diff(p) <= 0)
        or np.any(stress <= 0)
        or np.any(np.diff(stress) < 0)
    ):
        raise ValueError(
            f"{name} must start at zero plastic strain with increasing strains "
            "and positive, nondecreasing stresses"
        )
    return tuple(tuple(map(float, row)) for row in knots)


@dataclass(frozen=True)
class ParaboloidalMaterial:
    """Configure pressure-dependent plasticity from tensile and compressive curves."""

    E: float = 3.76
    nu: float = 0.39
    plastic_poisson: float = 0.32
    tension_curve: tuple = ((0.0, 0.029),)
    compression_curve: tuple = ((0.0, 0.067),)

    def __post_init__(self):
        """Validate material constants and normalize the strength curves."""
        if not np.isfinite([self.E, self.nu, self.plastic_poisson]).all():
            raise ValueError("Material parameters must be finite")
        if (
            self.E <= 0
            or not -1 < self.nu < 0.5
            or not 0 <= self.plastic_poisson <= 0.5
        ):
            raise ValueError(
                "Require E > 0, -1 < nu < .5 and 0 <= plastic_poisson <= .5"
            )
        object.__setattr__(
            self, "tension_curve", _curve(self.tension_curve, "tension_curve")
        )
        object.__setattr__(
            self,
            "compression_curve",
            _curve(self.compression_curve, "compression_curve"),
        )

    @property
    def K(self):
        """Return the isotropic elastic bulk modulus."""
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        """Return the isotropic elastic shear modulus."""
        return self.E / (2 * (1 + self.nu))

    @property
    def alpha(self):
        """Return the plastic flow parameter determined by the plastic Poisson ratio."""
        return 4.5 * (1 - 2 * self.plastic_poisson) / (1 + self.plastic_poisson)

    @property
    def equivalence(self):
        """sqrt(1/(1 + 2 nu_p^2)), the factor of the equivalent plastic strain."""
        return 1 / np.sqrt(1 + 2 * self.plastic_poisson**2)

    @property
    def yield_stress(self):
        """Initial tensile yield stress: the solver's stress scale."""
        return self.tension_curve[0][1]

    associated = False  # g is never proportional to f: GMRES
    viscosity = 0.0  # rate independent
    beta = 0.0  # the fused pressure tangent reads u directly (stored unit deviator)
    model_name = "small_strain_paraboloidal_Tschoegl_Melro"

    def configuration(self):
        """Return serializable material parameters and hardening curves."""
        values = asdict(self)
        for key in ("tension_curve", "compression_curve"):
            values[key] = [list(row) for row in values[key]]
        return values

    def strengths(self, p, xp=np):
        """sigma_t, sigma_c and their slopes at equivalent plastic strain p."""
        return (
            *_interpolate(self.tension_curve, p, xp),
            *_interpolate(self.compression_curve, p, xp),
        )


def _interpolate(knots, p, xp):
    knots = np.asarray(knots)
    value = xp.full(p.shape, knots[0, 1], dtype=p.dtype)
    slope = xp.zeros_like(value)
    for (p0, s0), (p1, s1) in zip(knots[:-1], knots[1:]):
        rate = (s1 - s0) / (p1 - p0)
        inside = p >= p0
        value = xp.where(inside, s0 + rate * xp.minimum(p - p0, p1 - p0), value)
        slope = xp.where(inside & (p < p1), rate, slope)
    return value, slope


def _scalar_equation(gamma, q, m, p0, material, xp, partial=False):
    """f(gamma) and its total derivative; with partial, also df/dq_trial and df/dm_trial."""
    mu, K, alpha, root_k = material.mu, material.K, material.alpha, material.equivalence
    beta = 4 * alpha * alpha / 3
    psi, omega = 1 / (1 + 6 * mu * gamma), 1 / (1 + 2 * K * alpha * gamma)
    radius = xp.sqrt(6 * psi * psi * q * q + beta * omega * omega * m * m)
    p = p0 + gamma * root_k * radius
    st, st_slope, sc, sc_slope = material.strengths(p, xp)
    f = psi * psi * q * q - 3 * (st - sc) * omega * m - st * sc
    f_p = -3 * (st_slope - sc_slope) * omega * m - (st_slope * sc + st * sc_slope)
    dpsi, domega = -6 * mu * psi * psi, -2 * K * alpha * omega * omega
    safe = xp.where(radius > 0, radius, 1)
    d_radius = (6 * psi * dpsi * q * q + beta * omega * domega * m * m) / safe
    df = (
        2 * psi * dpsi * q * q
        - 3 * (st - sc) * domega * m
        + f_p * root_k * (radius + gamma * d_radius)
    )
    if not partial:
        return f, df
    f_q = 2 * psi * psi * q + f_p * gamma * root_k * 6 * psi * psi * q / safe
    f_m = (
        -3 * (st - sc) * omega + f_p * gamma * root_k * beta * omega * omega * m / safe
    )
    return f, df, f_q, f_m, psi, omega, dpsi, domega, p


def _multiplier(q, m, p0, material, plastic, xp):
    """Root of f(gamma) = 0 on plastic voxels: Newton kept inside a sign bracket."""
    lo = xp.zeros_like(q)
    hi = xp.full_like(q, 1e-3 / material.mu)
    for _ in range(64):
        f, _ = _scalar_equation(hi, q, m, p0, material, xp)
        grow = plastic & (f >= 0)
        if not bool(xp.any(grow)):
            break
        lo = xp.where(grow, hi, lo)
        hi = xp.where(grow, 4 * hi, hi)
    x = lo.copy()
    eps = np.finfo(q.dtype).eps
    for _ in range(100):
        f, df = _scalar_equation(x, q, m, p0, material, xp)
        lo = xp.where(f > 0, x, lo)
        hi = xp.where(f > 0, hi, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - f / df
        inside = xp.isfinite(newton) & (newton > lo) & (newton < hi)
        new = xp.where(inside, newton, (lo + hi) / 2)
        done = xp.abs(new - x) <= 4 * eps * new
        x = xp.where(plastic, new, 0)
        if bool(xp.all(done | ~plastic)):
            break
    return x


def update_cpu(strain, state, material, ids, table, start=None):
    """Array reference of the backward Euler return map and its exact tangent."""
    if start is not None:
        raise ValueError("The paraboloidal model uses backward_euler")
    mu, K = material.mu, material.K
    ee = strain - state.plastic_strain
    trace = ee[:3].sum(axis=0)
    trial = 2 * mu * ee
    trial[:3] -= (2 * mu / 3) * trace
    norm = np.sqrt(np.sum(trial * trial, axis=0))
    q, m = np.sqrt(1.5) * norm, K * trace
    st, _, sc, _ = material.strengths(state.equivalent)
    plastic = (ids == 0) & (q * q - 3 * (st - sc) * m - st * sc > 0)
    gamma = _multiplier(q, m, state.equivalent, material, plastic, np)
    f, df, f_q, f_m, psi, omega, dpsi, domega, p = _scalar_equation(
        gamma, q, m, state.equivalent, material, np, partial=True
    )
    u = np.where(plastic, trial / np.maximum(norm, np.finfo(strain.dtype).tiny), 0)
    stress = psi * trial
    stress[:3] += omega * m
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(plastic, -1 / df, 0)
    # Derivatives of the multiplier: dq_trial = sqrt(6) mu u.de, dm_trial = K tr(de).
    to_u, to_trace = G * f_q * np.sqrt(6) * mu, G * f_m * K
    a = np.where(plastic, 2 * mu * psi, 2 * mu)
    b = np.where(plastic, dpsi * norm * to_u, 0)
    coupling_to_direction = np.where(plastic, dpsi * norm * to_trace, 0)
    coupling_to_pressure = np.where(plastic, domega * m * to_u, 0)
    bulk = np.where(plastic, K * omega + domega * m * to_trace, K)
    dp = np.where(plastic, p - state.equivalent, 0)
    if np.any(ids):
        fiber = elastic_action(strain, ids, table)
        stress[:, ids != 0] = fiber[:, ids != 0]
    return MaterialResponse(
        stress, u, a, b, dp, bulk, coupling_to_direction, coupling_to_pressure
    )


def tangent_cpu(value, response, material, ids, table):
    """Apply the host algorithmic tangent including volumetric couplings."""
    trace = np.sum(value[:3], axis=0)
    u = response.direction
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


def plastic_increment(stress, dp, material, xp=np):
    """Compute dep = gamma (3 s + 2/3 alpha m I) from the accepted stress."""
    m = xp.sum(stress[:3], axis=0) / 3
    s = stress.copy()
    s[:3] -= m
    q2 = 1.5 * xp.sum(s * s, axis=0)
    radius = xp.sqrt(6 * q2 + (4 * material.alpha**2 / 3) * m * m)
    gamma = xp.where(
        dp > 0, dp / (material.equivalence * xp.where(radius > 0, radius, 1)), 0
    )
    increment = 3 * gamma * s
    increment[:3] += (2 * material.alpha / 3) * gamma * m
    return increment


def commit_cpu(state, response, material):
    """Return a new committed plastic state from the accepted trial response."""
    return PlasticState(
        state.plastic_strain
        + plastic_increment(response.stress, response.dp, material),
        state.equivalent + response.dp,
    )


def audit_cpu(response, state, material, ids, xp=np):
    """Yield violation in stress units, q - sqrt(3 (st - sc) m + st sc), and plastic work."""
    m = xp.sum(response.stress[:3], axis=0) / 3
    s = response.stress.copy()
    s[:3] -= m
    q = xp.sqrt(1.5 * xp.sum(s * s, axis=0))
    st, _, sc, _ = material.strengths(state.equivalent + response.dp, xp)
    radicand = 3 * (st - sc) * m + st * sc
    allowed = xp.sign(radicand) * xp.sqrt(xp.abs(radicand))
    violation = xp.where(ids == 0, xp.maximum(q - allowed, 0), 0)
    work = xp.sum(
        response.stress * plastic_increment(response.stress, response.dp, material, xp),
        axis=0,
    )
    return violation, xp.where(ids == 0, work, 0)


_CUDA = r"""
__device__ __forceinline__ void curve(const double* knots,int count,REAL p,REAL& value,REAL& slope){
 value=L(knots[1]);slope=0;
 for(int i=0;i+1<count;i++){
   const REAL p0=L(knots[2*i]),p1=L(knots[2*i+2]),s0=L(knots[2*i+1]),s1=L(knots[2*i+3]);
   const REAL rate=(s1-s0)/(p1-p0);
   if(p>=p0){value=s0+rate*fmin(p-p0,p1-p0);slope=p<p1?rate:(REAL)0;}
 }
}
struct Scalar{REAL f,df,fq,fm,psi,omega,dpsi,domega,p;};
__device__ Scalar equation(REAL g,REAL q,REAL m,REAL p0,REAL mu,REAL K,REAL alpha,REAL rk,
 const double* ct,int nt,const double* cc,int nc){
 Scalar r;const REAL beta=4*alpha*alpha/3;
 r.psi=1/(1+6*mu*g);r.omega=1/(1+2*K*alpha*g);
 const REAL R=sqrt(6*r.psi*r.psi*q*q+beta*r.omega*r.omega*m*m);
 r.p=p0+g*rk*R;
 REAL st,dst,sc,dsc;curve(ct,nt,r.p,st,dst);curve(cc,nc,r.p,sc,dsc);
 r.f=r.psi*r.psi*q*q-3*(st-sc)*r.omega*m-st*sc;
 const REAL fp=-3*(dst-dsc)*r.omega*m-(dst*sc+st*dsc);
 r.dpsi=-6*mu*r.psi*r.psi;r.domega=-2*K*alpha*r.omega*r.omega;
 const REAL safe=R>0?R:(REAL)1;
 const REAL dR=(6*r.psi*r.dpsi*q*q+beta*r.omega*r.domega*m*m)/safe;
 r.df=2*r.psi*r.dpsi*q*q-3*(st-sc)*r.domega*m+fp*rk*(R+g*dR);
 r.fq=2*r.psi*r.psi*q+fp*g*rk*6*r.psi*r.psi*q/safe;
 r.fm=-3*(st-sc)*r.omega+fp*g*rk*beta*r.omega*r.omega*m/safe;
 return r;
}
extern "C" __global__ void pb_update(const MEM* e,const MEM* ep,const MEM* k,
 const int* ids,const MEM* table,MEM* sig,MEM* u,MEM* aa,MEM* bb,MEM* dp,
 MEM* bulk,MEM* cd,MEM* cp,long long n,PARAM mu_,PARAM K_,PARAM alpha_,PARAM rk_,
 const double* ct,int nt,const double* cc,int nc,const MEM* gd){
 const REAL mu=L(mu_),K=L(K_),alpha=L(alpha_),rk=L(rk_);
 // gd: optional damage degradation of the matrix (effective-stress coupling);
 // stress and compact tangent at fixed damage scale by it.
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){
   for(int i=0;i<6;i++){REAL v=0;for(int j=0;j<6;j++)v+=L(table[ids[z]*36+i*6+j])*L(e[j*n+z]);sig[i*n+z]=S(v);u[i*n+z]=S(0);}
   aa[z]=S(2*mu);bb[z]=S(0);dp[z]=S(0);bulk[z]=S(K);cd[z]=S(0);cp[z]=S(0);return;
 }
 REAL tr=0,s[6],n2=0;
 for(int i=0;i<3;i++)tr+=L(e[i*n+z])-L(ep[i*n+z]);
 for(int i=0;i<6;i++){s[i]=2*mu*(L(e[i*n+z])-L(ep[i*n+z])-(i<3?tr/3:0));n2+=s[i]*s[i];}
 const REAL norm=sqrt(n2),q=sqrt((REAL)1.5)*norm,m=K*tr,p0=L(k[z]);
 REAL st,dst,sc,dsc;curve(ct,nt,p0,st,dst);curve(cc,nc,p0,sc,dsc);
 const REAL gz=gd?L(gd[z]):(REAL)1;
 if(!(q*q-3*(st-sc)*m-st*sc>0)){
   for(int i=0;i<6;i++){sig[i*n+z]=S(gz*(s[i]+(i<3?m:0)));u[i*n+z]=S(0);}
   aa[z]=S(gz*2*mu);bb[z]=S(0);dp[z]=S(0);bulk[z]=S(gz*K);cd[z]=S(0);cp[z]=S(0);return;
 }
 // Bracket the root by growing gamma, then Newton steps kept inside it.
 REAL lo=0,hi=(REAL)1e-3/mu;
 for(int i=0;i<64&&equation(hi,q,m,p0,mu,K,alpha,rk,ct,nt,cc,nc).f>=0;i++){lo=hi;hi*=4;}
 REAL g=lo;Scalar r;
 for(int it=0;it<100;it++){
   r=equation(g,q,m,p0,mu,K,alpha,rk,ct,nt,cc,nc);
   if(r.f>0)lo=g;else hi=g;
   REAL next=g-r.f/r.df;
   if(!(isfinite(next)&&next>lo&&next<hi))next=(lo+hi)/2;
   const bool done=fabs(next-g)<=4*EPS*next;g=next;
   if(done)break;
 }
 r=equation(g,q,m,p0,mu,K,alpha,rk,ct,nt,cc,nc);
 const REAL G=-1/r.df,tu=G*r.fq*sqrt((REAL)6)*mu,tt=G*r.fm*K;
 aa[z]=S(gz*(2*mu*r.psi));bb[z]=S(gz*(r.dpsi*norm*tu));cd[z]=S(gz*(r.dpsi*norm*tt));cp[z]=S(gz*(r.domega*m*tu));
 bulk[z]=S(gz*(K*r.omega+r.domega*m*tt));dp[z]=S(r.p-p0);
 for(int i=0;i<6;i++){u[i*n+z]=S(norm>0?s[i]/norm:0);sig[i*n+z]=S(gz*(r.psi*s[i]+(i<3?r.omega*m:0)));}
}
"""
_CUDA_REST = r"""
extern "C" __global__ void pb_tangent(const REAL* v,const REAL* u,const REAL* a,const REAL* b,
 const REAL* bulk,const REAL* cd,const REAL* cp,const int* ids,const REAL* table,REAL* out,long long n){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){for(int i=0;i<6;i++){REAL w=0;for(int j=0;j<6;j++)w+=table[ids[z]*36+i*6+j]*v[j*n+z];out[i*n+z]=w;}return;}
 REAL tr=0,d=0;
 for(int i=0;i<6;i++){d+=u[i*n+z]*v[i*n+z];if(i<3)tr+=v[i*n+z];}
 for(int i=0;i<6;i++)
   out[i*n+z]=a[z]*v[i*n+z]+b[z]*u[i*n+z]*d+cd[z]*u[i*n+z]*tr+(i<3?(bulk[z]-a[z]/3)*tr+cp[z]*d:0);
}
// gamma from dp and the accepted stress; dep = gamma (3 s + 2/3 alpha m I).
__device__ __forceinline__ REAL multiplier(const REAL* sig,long long z,long long n,REAL dp,REAL alpha,REAL rk,REAL* s,REAL& m){
 m=(sig[z]+sig[n+z]+sig[2*n+z])/3;REAL q2=0;
 for(int i=0;i<6;i++){s[i]=sig[i*n+z]-(i<3?m:0);q2+=s[i]*s[i];}
 const REAL R=sqrt(9*q2+4*alpha*alpha*m*m/3);
 return dp>0&&R>0?dp/(rk*R):(REAL)0;
}
extern "C" __global__ void pb_commit(const REAL* ep,const REAL* k,const REAL* sig,const REAL* dp,
 REAL* ep_out,REAL* k_out,long long n,REAL alpha,REAL rk){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 REAL s[6],m;const REAL g=multiplier(sig,z,n,dp[z],alpha,rk,s,m);
 for(int i=0;i<6;i++)ep_out[i*n+z]=ep[i*n+z]+g*(3*s[i]+(i<3?2*alpha*m/3:0));
 k_out[z]=k[z]+dp[z];
}
extern "C" __global__ void pb_audit(const REAL* sig,const REAL* dp,const REAL* k,const int* ids,
 REAL* violation,REAL* work,REAL* volume,long long n,REAL alpha,REAL rk,
 const double* ct,int nt,const double* cc,int nc,const REAL* gd){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){violation[z]=0;work[z]=0;volume[z]=0;return;}
 REAL s[6],m;const REAL g=multiplier(sig,z,n,dp[z],alpha,rk,s,m);
 REAL q2=0;for(int i=0;i<6;i++)q2+=s[i]*s[i];
 REAL st,dst,sc,dsc;const REAL p=k[z]+dp[z];curve(ct,nt,p,st,dst);curve(cc,nc,p,sc,dsc);
 // Yield holds for the effective stress; work and volume are scale-free in g.
 const REAL gz=gd?gd[z]:(REAL)1;
 const REAL rad=3*(st-sc)*m/gz+st*sc,allowed=rad>=0?sqrt(rad):-sqrt(-rad);
 const REAL f=sqrt((REAL)1.5*q2)/gz-allowed;
 violation[z]=isnan(f)?f:fmax(f,(REAL)0);
 // sigma:dep = gamma (3 s:s + 2 alpha m^2) >= 0; volume = tr(dep).
 work[z]=g*(3*q2+2*alpha*m*m);volume[z]=2*alpha*g*m;
}
"""


class CuPyParaboloidal:
    """One CUDA thread per voxel; compact nonsymmetric tangent, elastic fibers."""

    def __init__(self, cp, material, ids, table, dtype, arithmetic="ieee"):
        self.cp, self.material, self.ids, self.table = cp, material, ids, table
        self.dtype = np.dtype(dtype).type
        self.shape, self.n = ids.shape, ids.size
        if arithmetic not in ("ieee", "df64") or (
            arithmetic == "df64" and self.dtype != np.float64
        ):
            raise ValueError(
                "arithmetic must be 'ieee', or 'df64' with float64 storage"
            )
        real = "float" if self.dtype == np.float32 else "double"
        identity = "#define MEM REAL\n#define PARAM REAL\n#define L(x) ((REAL)(x))\n#define S(x) (x)\n"
        code = (
            (identity + _CUDA + _CUDA_REST)
            .replace("REAL", real)
            .replace("EPS", repr(float(np.finfo(self.dtype).eps)))
        )
        mod = cp.RawModule(code=code)
        self.update_kernel, self.tangent_kernel = (
            mod.get_function("pb_update"),
            mod.get_function("pb_tangent"),
        )
        if arithmetic == "df64":
            # Only the return map is arithmetic-bound; tangent, commit and
            # audit stay IEEE FP64. Its bracketed Newton stops at df64 resolution.
            from cufhom.materials.double_float import DF64_EPSILON, df64_source

            self.update_kernel = cp.RawModule(
                code=df64_source(_CUDA).replace("EPS", f"df({DF64_EPSILON!r}f)")
            ).get_function("pb_update")
        self.commit_kernel, self.audit_kernel = (
            mod.get_function("pb_commit"),
            mod.get_function("pb_audit"),
        )
        self.curves = (
            cp.asarray(material.tension_curve, dtype=np.float64).ravel(),
            np.int32(len(material.tension_curve)),
            cp.asarray(material.compression_curve, dtype=np.float64).ravel(),
            np.int32(len(material.compression_curve)),
        )

    def _launch(self, kernel, args):
        kernel(((self.n + 255) // 256,), (256,), args)

    def update(self, strain, state, start=None):
        """Evaluate the device backward-Euler return map without committing state."""
        if start is not None:
            raise ValueError("The paraboloidal model uses backward_euler")
        cp, m, d = self.cp, self.material, self.dtype
        fields = [cp.empty(self.shape, dtype=d) for _ in range(6)]
        r = MaterialResponse(
            cp.empty_like(strain), cp.empty_like(strain), *fields[:3], *fields[3:]
        )
        self._launch(
            self.update_kernel,
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
                d(m.alpha),
                d(m.equivalence),
            )
            + self.curves
            + (self._degradation(),),
        )
        return r

    def _degradation(self):
        """Damage degradation field, or a null pointer (no damage)."""
        return np.uint64(0)

    def tangent(self, value, response):
        """Apply the device algorithmic tangent to a strain increment."""
        out = self.cp.empty_like(value)
        self._launch(
            self.tangent_kernel,
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
            ),
        )
        return out

    def commit(self, state, response):
        """Return a new committed device plastic state from an accepted response."""
        ep, k = (
            self.cp.empty_like(state.plastic_strain),
            self.cp.empty_like(state.equivalent),
        )
        self._launch(
            self.commit_kernel,
            (
                state.plastic_strain,
                state.equivalent,
                response.stress,
                response.dp,
                ep,
                k,
                np.int64(self.n),
                self.dtype(self.material.alpha),
                self.dtype(self.material.equivalence),
            ),
        )
        return PlasticState(ep, k)

    def audit(self, response, state):
        """Yield violation, plastic work and plastic volume increment per voxel."""
        violation, work, volume = (
            self.cp.empty(self.shape, dtype=self.dtype) for _ in range(3)
        )
        self._launch(
            self.audit_kernel,
            (
                response.stress,
                response.dp,
                state.equivalent,
                self.ids,
                violation,
                work,
                volume,
                np.int64(self.n),
                self.dtype(self.material.alpha),
                self.dtype(self.material.equivalence),
            )
            + self.curves
            + (self._degradation(),),
        )
        return violation, work, volume
