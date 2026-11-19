"""Small-strain Mandel J2 elasto-(visco)plasticity (stress units: GPa).

The hardening laws are those of the FANS small-strain J2 models: isotropic
R(p) = yield_stress + hardening*p + (saturation_stress - yield_stress)
(1 - exp(-saturation_exponent*p)), linear (Prager) kinematic hardening with
backstress X = 2/3 kinematic_hardening dev(plastic_strain), and a Perzyna
overstress q(sigma - X) - R(p) = 3/2 viscosity dp/dt. hardening = 0 without
saturation is perfect plasticity. X is a function of the plastic strain, so
no state beyond (plastic_strain, equivalent) is stored. Every combination
keeps the compact tangent a, b, u below; a nonlinear R(p) adds a scalar
Newton solve per voxel, while linear laws keep the closed-form returns.

The state is immutable during equilibrium iterations. A response contains only
the compact consistent tangent and the plastic increment needed for commit.
The CuPy kernels follow exactly the CPU formulas, one thread per voxel.

Two return maps. Backward Euler (radial return) flows along the trial
deviator. The generalized midpoint rule (Ortiz & Popov 1985), selected by
passing the strain at the start of the step, flows along the deviator of
(sigma_n + sigma_n+1)/2, which is parallel to s_n + s_trial; with yield
enforced at n+1 the increment is the smaller root of a quadratic. It is second
order in the load step and equals radial return for proportional paths. Its
exact tangent is nonsymmetric; the response keeps the symmetric part along
the flow direction (a, b, u), so Newton still uses CG and the residual and
acceptance remain exact. Where the quadratic has no admissible root (a load
reversal crossing the whole elastic domain in one step) radial return is used.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class J2Material:
    """Configure small-strain J2 plasticity with optional hardening and viscosity."""

    E: float = 3.0
    nu: float = 0.3
    yield_stress: float = 0.015
    hardening: float = 0.03
    saturation_stress: float | None = None
    saturation_exponent: float = 0.0
    kinematic_hardening: float = 0.0
    viscosity: float = 0.0

    def __post_init__(self):
        """Validate elastic, yield, hardening and viscosity parameters."""
        values = (
            self.E,
            self.nu,
            self.yield_stress,
            self.hardening,
            self.saturation_exponent,
            self.kinematic_hardening,
            self.viscosity,
        )
        if not all(np.isfinite(v) for v in values) or not np.isfinite(
            self.saturation_stress or 0.0
        ):
            raise ValueError("Material parameters must be finite")
        if (
            self.E <= 0
            or not -1 < self.nu < 0.5
            or self.yield_stress <= 0
            or self.hardening < 0
        ):
            raise ValueError(
                "Require E, yield_stress > 0, hardening >= 0 and -1 < nu < .5"
            )
        if min(self.kinematic_hardening, self.viscosity) < 0:
            raise ValueError("Kinematic hardening and viscosity must be nonnegative")
        if self.saturation_stress is None:
            if self.saturation_exponent:
                raise ValueError("saturation_exponent requires saturation_stress")
        elif (
            self.saturation_stress < self.yield_stress or self.saturation_exponent <= 0
        ):
            raise ValueError(
                "Require saturation_stress >= yield_stress and saturation_exponent > 0"
            )

    @property
    def saturation(self):
        """Amplitude of the exponential isotropic hardening (0 when absent)."""
        return (
            0.0
            if self.saturation_stress is None
            else self.saturation_stress - self.yield_stress
        )

    @property
    def nonlinear_hardening(self):
        """Return whether a nonzero saturating hardening contribution is active."""
        return self.saturation != 0

    @property
    def general(self):
        """Kinematic, saturating or viscous: the iterative, arithmetic-heavy return."""
        return bool(
            self.kinematic_hardening or self.nonlinear_hardening or self.viscosity
        )

    def radius(self, p, xp=np):
        """Isotropic yield radius R(p); bitwise yield_stress + hardening*p when linear."""
        radius = self.yield_stress + self.hardening * p
        if self.nonlinear_hardening:
            radius = radius + self.saturation * (
                1 - xp.exp(-self.saturation_exponent * p)
            )
        return radius

    def slope(self, p, xp=np):
        """dR/dp."""
        if not self.nonlinear_hardening:
            return self.hardening
        return self.hardening + self.saturation * self.saturation_exponent * xp.exp(
            -self.saturation_exponent * p
        )

    def backstress(self, plastic_strain, xp=np):
        """Prager backstress 2/3 kinematic_hardening dev(plastic_strain), or None."""
        if not self.kinematic_hardening:
            return None
        X = (2 * self.kinematic_hardening / 3) * plastic_strain
        X[:3] -= xp.sum(X[:3], axis=0) / 3
        return X

    def viscous_modulus(self, dt):
        """Overstress per unit dp, 3/2 viscosity/dt (FANS eta/dt in q units)."""
        if not self.viscosity:
            return 0.0
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Viscoplasticity needs a positive finite time increment")
        return 1.5 * self.viscosity / dt

    @property
    def model_name(self):
        """Return an identifier describing the active constitutive mechanisms."""
        if not (self.nonlinear_hardening or self.kinematic_hardening or self.viscosity):
            return "small_strain_J2_linear_isotropic_hardening"
        name = (
            "small_strain_J2_"
            + ("Voce" if self.nonlinear_hardening else "linear")
            + "_isotropic"
        )
        if self.kinematic_hardening:
            name += "_Prager_kinematic"
        return name + ("_Perzyna_viscoplastic" if self.viscosity else "_hardening")

    def configuration(self):
        """Parameters; default extensions are omitted (stable legacy fingerprints)."""
        values = asdict(self)
        for key, default in (
            ("saturation_stress", None),
            ("saturation_exponent", 0.0),
            ("kinematic_hardening", 0.0),
            ("viscosity", 0.0),
        ):
            if values.get(key, default) == default:
                values.pop(key, None)
        return values

    @property
    def K(self):
        """Return the isotropic elastic bulk modulus."""
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        """Return the isotropic elastic shear modulus."""
        return self.E / (2 * (1 + self.nu))


@dataclass
class PlasticState:
    """Store committed Mandel plastic strain and accumulated equivalent strain."""

    plastic_strain: object
    equivalent: object


@dataclass
class MaterialResponse:
    """Store trial stress, plastic increment and compact tangent coefficients."""

    stress: object
    direction: object
    a: object
    b: object
    dp: object
    # Optional compact pressure-dependent tangent. None keeps the J2 law.
    bulk: object = None
    coupling_to_direction: object = None
    coupling_to_pressure: object = None


def initial_state(shape, xp=np, dtype=np.float64):
    """Create an undeformed plastic state on the selected array backend."""
    return PlasticState(
        xp.zeros((6,) + tuple(shape), dtype=dtype), xp.zeros(shape, dtype=dtype)
    )


def elastic_action(value, ids, table):
    """CPU indexed Mandel matrix multiplication, without a dense C(x)."""
    out = np.zeros_like(value)
    for i in range(6):
        for j in range(6):
            out[i] += table[ids, i, j] * value[j]
    return out


def update_cpu(strain, state, material, ids, table, start=None, viscous=0.0):
    """Return map; viscous = material.viscous_modulus(dt) for viscoplasticity."""
    mu, K = material.mu, material.K
    kinematic = material.kinematic_hardening
    ee = strain - state.plastic_strain
    trace = ee[:3].sum(axis=0)
    trial = 2 * mu * ee
    trial[:3] -= (2 * mu / 3) * trace
    backstress = material.backstress(state.plastic_strain)
    relative = trial if backstress is None else trial - backstress
    norm = np.sqrt(np.sum(relative * relative, axis=0))
    q = np.sqrt(1.5) * norm
    radius = material.radius(state.equivalent)
    hardening = material.hardening + viscous
    dp = np.maximum(q - radius, 0)
    dp /= 3 * mu + kinematic + hardening
    dp = np.where(ids == 0, dp, 0)
    if material.nonlinear_hardening:
        dp = _radial_newton_cpu(q, state.equivalent, dp, material, viscous)
    direction = relative / np.maximum(norm, np.finfo(strain.dtype).tiny)
    # q_new = radius + H*dp. This equals q - 3*mu*dp without
    # subtracting nearly equal stresses in the small-hardening limit.
    factor = np.ones_like(q)
    if material.nonlinear_hardening:
        returned = material.radius(state.equivalent + dp) + (kinematic + viscous) * dp
        hardening = kinematic + viscous + material.slope(state.equivalent + dp)
    else:
        returned = radius + (kinematic + hardening) * dp
        hardening = kinematic + hardening
    np.divide(returned, q, out=factor, where=dp > 0)
    a = 2 * mu * factor
    b = np.where(dp > 0, 2 * mu * hardening / (3 * mu + hardening) - a, 0)
    stress = trial * (a / (2 * mu))
    if backstress is not None:
        stress += (1 - a / (2 * mu)) * backstress
    stress[:3] += K * trace
    if start is not None:
        _midpoint_cpu(
            strain, start, state, material, dp, direction, a, b, stress, viscous
        )
    if np.any(ids):
        fiber_stress = elastic_action(strain, ids, table)
        stress[:, ids != 0] = fiber_stress[:, ids != 0]
    return MaterialResponse(stress, direction, a, b, dp)


def _radial_newton_cpu(q, alpha, dp, material, viscous):
    """Scalar Newton on q - (3mu+C+V)dp - R(alpha+dp) = 0 from the linearized root.

    The residual is convex and decreasing (R is concave), so the iterates
    increase monotonically to the root.
    """
    linear = 3 * material.mu + material.kinematic_hardening + viscous
    d = np.where(
        dp > 0, (q - material.radius(alpha)) / (linear + material.slope(alpha)), 0
    )
    for _ in range(60):
        g = q - linear * d - material.radius(alpha + d)
        step = g / (linear + material.slope(alpha + d))
        d = np.where(dp > 0, d + step, 0)
        if not np.any(np.abs(step) > 4 * np.finfo(q.dtype).eps * np.abs(d)):
            break
    return d


def _midpoint_cpu(
    strain, start, state, material, dp, direction, a, b, stress, viscous=0.0
):
    """Overwrite plastic voxels with the midpoint rule, in place, in float64 (see the CUDA update)."""
    mu, K, H, sy = material.mu, material.K, material.hardening, material.yield_stress
    C, V = material.kinematic_hardening, viscous
    if np.dtype(strain.dtype) == np.float32:  # the kernel's parameters are float32 too
        mu, K, H, sy, C, V = (float(np.float32(v)) for v in (mu, K, H, sy, C, V))
    ee = strain.astype(np.float64) - state.plastic_strain
    trace = ee[:3].sum(axis=0)
    trial = 2 * mu * ee
    trial[:3] -= (2 * mu / 3) * trace
    e0 = start.astype(np.float64) - state.plastic_strain
    m = trial + 2 * mu * e0
    m[:3] -= (2 * mu / 3) * e0[:3].sum(axis=0)
    relative = trial
    if C:
        # Midpoint flow along xi_n + xi_trial, xi = s - X with X fixed at n.
        X = (2 * C / 3) * state.plastic_strain.astype(np.float64)
        X[:3] -= X[:3].sum(axis=0) / 3
        relative = trial - X
        m -= 2 * X
    mn = np.sqrt(np.sum(m * m, axis=0))
    m /= np.maximum(mn, np.finfo(np.float64).tiny)
    A = np.sum(relative * m, axis=0)
    q = np.sqrt(1.5 * np.sum(relative * relative, axis=0))
    # c: decrease of xi per unit dp; cs: decrease of the stress itself.
    mk = mu + C / 3
    c, cs = np.sqrt(6) * mk, np.sqrt(6) * mu
    alpha = state.equivalent.astype(np.float64)
    if material.nonlinear_hardening:
        R0 = material.radius(alpha)
        H = V + material.slope(alpha)
    else:
        R0 = sy + H * alpha
        H = H + V
    a2, a1, a0 = 9 * mk * mk - H * H, 3 * c * A + 2 * R0 * H, (q - R0) * (q + R0)
    disc = a1 * a1 - 4 * a2 * a0
    mid = (dp > 0) & (A > 0) & (disc >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 2 * a0 / (a1 + np.sqrt(np.where(mid, disc, 0)))
        if material.nonlinear_hardening:
            d, D = _midpoint_newton_cpu(
                relative, m, c, alpha, np.where(mid, d, 0), material, V
            )
        else:
            D = 2 * a2 * d - a1
        al1 = 1 - cs * d / mn
        kq = 1 - cs * (3 * c * d - 3 * A) / D
        full = trial - cs * d * m
    # Working-precision and FP64 branch tests can straddle first yield.
    # An inadmissible root must keep the nonnegative radial return above.
    mid &= np.isfinite(d) & (d >= 0)
    dp[mid] = d[mid]
    a[mid] = 2 * mu * al1[mid]
    b[mid] = 2 * mu * (kq - al1)[mid]
    direction[:, mid] = m[:, mid]
    full[:3] += K * trace
    stress[:, mid] = full[:, mid]


def _midpoint_newton_cpu(relative, m, c, alpha, d, material, viscous):
    """Newton on sqrt(1.5)|xi_trial - c d m| = R(alpha+d) + V d from the linearized root.

    Returns d and D = dF/dd of F = 1.5|xi - c d m|^2 - Y(d)^2 (tangent term).
    """
    r15 = np.sqrt(1.5)
    for _ in range(60):
        t = relative - c * d * m
        sn = np.sqrt(np.sum(t * t, axis=0))
        sm = np.sum(t * m, axis=0)
        Y = material.radius(alpha + d) + viscous * d
        Yp = material.slope(alpha + d) + viscous
        step = (r15 * sn - Y) / (-r15 * c * sm / sn - Yp)
        d = d - step
        if not np.any(np.abs(step) > 4 * np.finfo(np.float64).eps * np.abs(d)):
            break
    A = np.sum(relative * m, axis=0)
    Y = material.radius(alpha + d) + viscous * d
    return d, 3 * c * (c * d - A) - 2 * Y * (material.slope(alpha + d) + viscous)


def tangent_cpu(value, response, material, ids, table):
    """Apply the algorithmic J2 tangent, retaining indexed elastic fiber behavior."""
    trace = value[:3].sum(axis=0)
    out = response.a * value + response.b * response.direction * np.sum(
        response.direction * value, axis=0
    )
    out[:3] += (material.K - response.a / 3) * trace
    if np.any(ids):
        fiber = elastic_action(value, ids, table)
        out[:, ids != 0] = fiber[:, ids != 0]
    return out


def commit_state(state, response):
    """Allocate a new state; neither the response nor the old state is mutated."""
    return PlasticState(
        state.plastic_strain + np.sqrt(1.5) * response.direction * response.dp,
        state.equivalent + response.dp,
    )


_CUDA = r"""
// GENERAL selects kinematic/saturating/viscous laws; without it the code is
// the linear isotropic return map (same operations, bitwise results).
// Isotropic radius R(p) and slope; NONLINEAR adds the saturation (Voce) term.
__device__ __forceinline__ REAL radius(REAL p,REAL sy,REAL H,REAL Q,REAL dl){
 REAL r=sy+H*p;
#if NONLINEAR
 r+=Q*(1-exp(-dl*p));
#endif
 return r;}
__device__ __forceinline__ REAL slope(REAL p,REAL H,REAL Q,REAL dl){
#if NONLINEAR
 return H+Q*dl*exp(-dl*p);
#else
 return H;
#endif
}
extern "C" __global__ void update(const MEM* e, const MEM* ep, const MEM* alpha,
 const int* ids, const MEM* table, MEM* sig, MEM* u, MEM* a, MEM* b, MEM* dp,
 long long n, PARAM mu_, PARAM K_, PARAM sy_, PARAM H_, const MEM* e0,
 PARAM C_, PARAM Q_, PARAM dl_, PARAM V_, PARAM tol_) {
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return;
 const REAL mu=L(mu_), K=L(K_), sy=L(sy_), H=L(H_), C=L(C_), Q=L(Q_), dl=L(dl_), V=L(V_), tol=L(tol_);
 REAL s[6], tr=0, norm2=0;
 for(int i=0;i<3;i++)tr+=L(e[i*n+k])-L(ep[i*n+k]);
#if GENERAL
 // xi = s - X, Prager backstress X = 2/3 C dev(ep) (zero when C = 0).
 REAL x[6], pt=0;
 for(int i=0;i<3;i++)pt+=L(ep[i*n+k]);
 for(int i=0;i<6;i++){s[i]=2*mu*(L(e[i*n+k])-L(ep[i*n+k])-(i<3?tr/3:0)); x[i]=(2*C/3)*(L(ep[i*n+k])-(i<3?pt/3:0));
   norm2+=(s[i]-x[i])*(s[i]-x[i]);}
 const REAL radius0=radius(L(alpha[k]),sy,H,Q,dl), Hq=slope(L(alpha[k]),H,Q,dl)+V, mk=mu+C/3;
#define XI(i) (s[i]-x[i])
#define RADIUS0 radius0
#else
 // R0 stays inside each branch: hoisting it changes FMA contraction.
 for(int i=0;i<6;i++){s[i]=2*mu*(L(e[i*n+k])-L(ep[i*n+k])-(i<3?tr/3:0)); norm2+=s[i]*s[i];}
 const REAL Hq=H, mk=mu;
#define XI(i) s[i]
#define RADIUS0 (sy+H*L(alpha[k]))
#endif
 REAL norm=sqrt(norm2), q=sqrt((REAL)1.5)*norm;
 if(e0&&ids[k]==0&&q>RADIUS0) {
   // Midpoint rule: flow along the deviator of (xi_n+xi_n+1)/2, which is
   // parallel to xi_n+xi_trial; yield exactly at n+1 (a quadratic in dp,
   // linearized for a saturating law).
   REAL m[6], t0=0, mn2=0, A=0;
   for(int i=0;i<3;i++)t0+=L(e0[i*n+k])-L(ep[i*n+k]);
   for(int i=0;i<6;i++){m[i]=s[i]+2*mu*(L(e0[i*n+k])-L(ep[i*n+k])-(i<3?t0/3:0));
#if GENERAL
     m[i]-=2*x[i];
#endif
     mn2+=m[i]*m[i];}
   const REAL mn=sqrt(mn2);
   for(int i=0;i<6;i++){m[i]/=mn; A+=XI(i)*m[i];}
   // c: decrease of xi per unit dp; cs: decrease of the stress.
   const REAL c=sqrt((REAL)6)*mk, cs=sqrt((REAL)6)*mu, R0=RADIUS0, a2=9*mk*mk-Hq*Hq, a1=3*c*A+2*R0*Hq;
   const REAL disc=a1*a1-4*a2*(q-R0)*(q+R0);
   if(A>0&&disc>=0) {
     REAL d=2*(q-R0)*(q+R0)/(a1+sqrt(disc));
     // The quadratic cancels for large local increments: one Newton step on
     // the yield condition, from the stress actually formed, restores it to
     // rounding (float32 kept the surface ~6x worse than radial return).
     // A saturating law iterates to convergence from that linearized root.
     const REAL r15=sqrt((REAL)1.5);
#if NONLINEAR
     REAL Y=R0, Yp=Hq;
     for(int it=0;it<32;it++){
       Y=radius(L(alpha[k])+d,sy,H,Q,dl)+V*d; Yp=slope(L(alpha[k])+d,H,Q,dl)+V;
#endif
       REAL sn2=0, sm=0;
       for(int i=0;i<6;i++){const REAL t=XI(i)-c*d*m[i]; sn2+=t*t; sm+=t*m[i];}
       const REAL sn=sqrt(sn2);
#if NONLINEAR
       const REAL step=(r15*sn-Y)/(-r15*c*sm/sn-Yp);
       d-=step;
       if(!(fabs(step)>tol*d))break;
     }
     Y=radius(L(alpha[k])+d,sy,H,Q,dl)+V*d; Yp=slope(L(alpha[k])+d,H,Q,dl)+V;
     const REAL D=3*c*(c*d-A)-2*Y*Yp;
#else
       d-=(r15*sn-R0-Hq*d)/(-r15*c*sm/sn-Hq);
     const REAL D=2*a2*d-a1;
#endif
     // Roundoff at first yield can make the Newton correction cross dp=0.
     // Keep the radial return if this is no longer an admissible root.
     if(isfinite(d)&&d>=0){
       const REAL al1=1-cs*d/mn, kq=1-cs*(3*c*d-3*A)/D;
       dp[k]=S(d); a[k]=S(2*mu*al1); b[k]=S(2*mu*(kq-al1));
       for(int i=0;i<6;i++){u[i*n+k]=S(m[i]); sig[i*n+k]=S(s[i]-cs*d*m[i]+(i<3?K*tr:0));}
       return;
     }
   }
 }
#if GENERAL
 const REAL R0=RADIUS0;
 REAL d=ids[k]==0?fmax((REAL)0,(q-R0)/(3*mu+C+Hq)):(REAL)0;
#if NONLINEAR
 // Monotone Newton on q-(3mu+C+V)d-R(alpha+d)=0 (R concave).
 if(d>0)for(int it=0;it<32;it++){
   const REAL step=(q-(3*mu+C+V)*d-radius(L(alpha[k])+d,sy,H,Q,dl))/(3*mu+C+V+slope(L(alpha[k])+d,H,Q,dl));
   d+=step; if(!(fabs(step)>tol*d))break;
 }
#endif
 // q-3*mu*d = R(alpha+d)+(C+V)d without subtracting nearly equal stresses.
 const REAL factor=d>0?(radius(L(alpha[k])+d,sy,H,Q,dl)+(C+V)*d)/q:(REAL)1;
 const REAL h=C+slope(L(alpha[k])+d,H,Q,dl)+V;
#else
 const REAL radius=RADIUS0;
 REAL d=ids[k]==0?fmax((REAL)0,(q-radius)/(3*mu+H)):(REAL)0;
 const REAL factor=d>0?(radius+H*d)/q:(REAL)1;
 const REAL h=H;
#endif
 dp[k]=S(d); a[k]=S(2*mu*factor);
 b[k]=S(d>0?2*mu*h/(3*mu+h)-L(a[k]):0);
 for(int i=0;i<6;i++){
   u[i*n+k]=S(norm>0?XI(i)/norm:0);
#if GENERAL
   REAL v=s[i]*(L(a[k])/(2*mu))+(1-L(a[k])/(2*mu))*x[i]+(i<3?K*tr:0);
#else
   REAL v=s[i]*(L(a[k])/(2*mu))+(i<3?K*tr:0);
#endif
   if(ids[k]){v=0; for(int j=0;j<6;j++)v+=L(table[ids[k]*36+i*6+j])*L(e[j*n+k]);}
   sig[i*n+k]=S(v);
 }
}
extern "C" __global__ void tangent(const MEM* v, const MEM* u, const MEM* a,
 const MEM* b, const int* ids, const MEM* table, MEM* out, long long n, PARAM K_){
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return;
 const REAL K=L(K_);
 REAL tr=L(v[k])+L(v[n+k])+L(v[2*n+k]), dot=0;
 for(int i=0;i<6;i++)dot+=L(u[i*n+k])*L(v[i*n+k]);
 for(int i=0;i<6;i++){
   REAL w=L(a[k])*L(v[i*n+k])+L(b[k])*L(u[i*n+k])*dot+(i<3?(K-L(a[k])/3)*tr:0);
   if(ids[k]){w=0;for(int j=0;j<6;j++)w+=L(table[ids[k]*36+i*6+j])*L(v[j*n+k]);}
   out[i*n+k]=S(w);
 }
}
"""


class CuPyMaterial:
    """arithmetic='df64' evaluates an FP64 law in double-float (see double_float)."""

    def __init__(self, cp, material, ids, table, dtype, arithmetic="ieee"):
        self.cp, self.material, self.ids, self.table = cp, material, ids, table
        self.dtype = np.dtype(dtype).type
        if arithmetic not in ("ieee", "df64") or (
            arithmetic == "df64" and self.dtype != np.float64
        ):
            raise ValueError(
                "arithmetic must be 'ieee', or 'df64' with float64 storage"
            )
        self.shape, self.n = ids.shape, ids.size
        m = material
        general = m.general
        defines = f"#define GENERAL {int(general)}\n#define NONLINEAR {int(m.nonlinear_hardening)}\n"
        if arithmetic == "df64":
            from cufhom.materials.double_float import df64_source

            code = defines + df64_source(_CUDA)
        else:
            real = "float" if self.dtype == np.float32 else "double"
            # Storage, parameters and arithmetic share one type here; L/S are
            # identities (the source expands to the same expressions).
            defines += f"#define MEM {real}\n#define PARAM {real}\n#define L(x) (x)\n#define S(x) (x)\n"
            code = defines + _CUDA.replace("REAL", real)
        mod = cp.RawModule(code=code)
        self.update_kernel, self.tangent_kernel = (
            mod.get_function("update"),
            mod.get_function("tangent"),
        )

    def update(self, strain, state, start=None, viscous=0.0):
        """Backward Euler, or the midpoint rule when the start-of-step strain is given.

        viscous = material.viscous_modulus(dt) for viscoplasticity.
        """
        cp, m, d = self.cp, self.material, self.dtype
        response = MaterialResponse(
            cp.empty_like(strain),
            cp.empty_like(strain),
            cp.empty(self.shape, dtype=d),
            cp.empty(self.shape, dtype=d),
            cp.empty(self.shape, dtype=d),
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
                response.stress,
                response.direction,
                response.a,
                response.b,
                response.dp,
                np.int64(self.n),
                d(m.mu),
                d(m.K),
                d(m.yield_stress),
                d(m.hardening),
                start if start is not None else np.uint64(0),
                d(m.kinematic_hardening),
                d(m.saturation),
                d(m.saturation_exponent),
                d(viscous),
                d(4 * np.finfo(d).eps),
            ),
        )
        return response

    def tangent(self, value, response):
        """Apply the cached device algorithmic tangent to a strain increment."""
        out = self.cp.empty_like(value)
        self.tangent_kernel(
            ((self.n + 255) // 256,),
            (256,),
            (
                value,
                response.direction,
                response.a,
                response.b,
                self.ids,
                self.table,
                out,
                np.int64(self.n),
                self.dtype(self.material.K),
            ),
        )
        return out
