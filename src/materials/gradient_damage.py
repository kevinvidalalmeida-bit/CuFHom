"""GPU effective-stress plasticity with irreversible implicit-gradient matrix damage.

The matrix law is Drucker-Prager or the paraboloidal polymer model (a
Melro-type plastic-damage matrix). The modified-von-Mises total-strain
source and exponential damage history are regularized by a periodic
Helmholtz equation. At fixed damage the exact plastic tangent is multiplied
by the degradation, including its volumetric terms. No dense tangent or
plastic trial-state field is materialized.

An optional fiber-matrix interphase (Sharma et al. 2020, Comput. Mech. 65,
doi:10.1007/s00466-019-01801-4: a voxel interphase with regularized damage
replaces the cohesive zone that an FFT grid cannot represent) is the matrix
within interphase_thickness of the fibers (city-block distance on the
grid). It keeps the matrix plasticity and nonlocal field, with its own
damage threshold and softening strain, i.e. interface strength and
toughness. Damage is isotropic in the interphase as in the bulk. These are
phenomenological demonstrations, not calibrated laws.
"""

from dataclasses import asdict, dataclass

import numpy as np
from cufhom.materials import drucker_prager as dp
from cufhom.materials import paraboloidal as pb
from cufhom.materials.plasticity import MaterialResponse


@dataclass(frozen=True)
class GradientDamageParameters:
    """Configure implicit gradient damage, regularization and residual stiffness."""

    model: str = "implicit_gradient"
    strain_threshold: float = 0.01
    softening_strain: float = 0.015
    length_scale: float = 0.03
    strength_ratio: float = 2.0
    residual_stiffness: float = 0.001
    interphase_thickness: float = 0.0
    interphase_strain_threshold: float | None = None
    interphase_softening_strain: float | None = None
    rtol: float = 1e-5
    max_iterations: int = 60
    helmholtz_rtol: float = 1e-6
    helmholtz_max_iterations: int = 500

    def __post_init__(self):
        """Validate damage, regularization and convergence parameters."""
        if self.model != "implicit_gradient":
            raise ValueError("damage.model must be implicit_gradient")
        values = [
            self.strain_threshold,
            self.softening_strain,
            self.length_scale,
            self.strength_ratio,
            self.residual_stiffness,
            self.rtol,
            self.helmholtz_rtol,
        ]
        if not np.isfinite(values).all():
            raise ValueError("Damage parameters must be finite")
        if min(self.strain_threshold, self.softening_strain, self.length_scale) <= 0:
            raise ValueError(
                "Damage threshold, softening strain and physical length must be positive"
            )
        if self.strength_ratio < 1 or not 0 < self.residual_stiffness < 1:
            raise ValueError(
                "Damage requires strength_ratio >= 1 and 0 < residual_stiffness < 1"
            )
        if not 0 < self.rtol < 1 or not 0 < self.helmholtz_rtol < 1:
            raise ValueError("Damage tolerances must lie in (0,1)")
        for value in (self.max_iterations, self.helmholtz_max_iterations):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("Damage iteration limits must be positive integers")
        interphase = (
            self.interphase_strain_threshold,
            self.interphase_softening_strain,
        )
        if not np.isfinite(self.interphase_thickness) or self.interphase_thickness < 0:
            raise ValueError("interphase_thickness must be finite and nonnegative")
        if self.interphase_thickness > 0:
            if (
                None in interphase
                or not np.isfinite(interphase).all()
                or min(interphase) <= 0
            ):
                raise ValueError(
                    "An interphase needs positive interphase threshold and softening strains"
                )
        elif interphase != (None, None):
            raise ValueError(
                "Interphase damage parameters require interphase_thickness > 0"
            )

    def configuration(self):
        """Parameters; an absent interphase is omitted (stable legacy fingerprints)."""
        values = asdict(self)
        if not self.interphase_thickness:
            for key in (
                "interphase_thickness",
                "interphase_strain_threshold",
                "interphase_softening_strain",
            ):
                values.pop(key)
        return values


def interphase_mask(cp, ids, lengths, thickness):
    """Matrix voxels within a physical city-block distance of the fibers."""
    steps = [max(1, int(round(thickness * n / L))) for n, L in zip(ids.shape, lengths)]
    fiber = ids != 0
    grown = fiber.copy()
    for step in range(max(steps)):
        previous = grown
        for axis, count in enumerate(steps):
            if step < count:
                grown = grown | cp.roll(previous, 1, axis) | cp.roll(previous, -1, axis)
    return (grown & ~fiber).astype(cp.uint8)


@dataclass(frozen=True)
class DamageState:
    """Store committed damage and its irreversible driving-history field."""

    value: object
    history: object


_EXTRA = r"""
extern "C" __global__ void gd_source(const REAL* e,const int* ids,double* out,
 long long n,double nu,double ratio){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){out[z]=0;return;}
 double tr=0,j2=0;for(int i=0;i<3;i++)tr+=(double)e[i*n+z];
 for(int i=0;i<6;i++){double v=(double)e[i*n+z]-(i<3?tr/3:0);j2+=.5*v*v;}
 const double t=(ratio-1)*tr/(1-2*nu);
 const double k=(t+sqrt(t*t+12*ratio*j2/((1+nu)*(1+nu))))/(2*ratio);
 out[z]=isfinite(k)?fmax(k,0.):k;
}
extern "C" __global__ void gd_history(const double* field,const double* hn,const REAL* dn,
 const int* ids,double* h,REAL* d,long long n,double km,double ksm,
 const unsigned char* interphase,double ki,double ksi){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){h[z]=0;d[z]=0;return;}
 const bool weak=interphase&&interphase[z];
 const double k0=weak?ki:km,ks=weak?ksi:ksm;
 const double v=fmax((double)hn[z],(double)field[z]);
 const double candidate=v>k0?1-k0/v*exp(-(v-k0)/ks):0;
 h[z]=v;d[z]=(REAL)fmax((double)dn[z],fmin(candidate,1.));
}
extern "C" __global__ void gd_audit(const REAL* sig,const REAL* gamma,const REAL* k,
 const REAL* g,const int* ids,REAL* violation,REAL* work,long long n,
 double sy,double H,double alpha,double beta){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){violation[z]=0;work[z]=0;return;}
 const double mean=((double)sig[z]+sig[n+z]+sig[2*n+z])/(3*g[z]);double q2=0;
 for(int i=0;i<6;i++){double s=(double)sig[i*n+z]/g[z]-(i<3?mean:0);q2+=s*s;}
 const double f=sqrt(1.5*q2)+alpha*mean-sy-H*((double)k[z]+gamma[z]);
 violation[z]=(REAL)(isnan(f)?f:fmax(f,0.));
 work[z]=(REAL)((double)g[z]*gamma[z]*(sy+(beta-alpha)*mean));
}
extern "C" __global__ void gd_damage_work(const REAL* e,const REAL* ep,const REAL* k,
 const REAL* u,const REAL* gamma,const REAL* d,const REAL* dn,const int* ids,
 double* out,long long n,double mu,double K,double H,double residual){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){out[z]=0;return;}
 double ee[6],tr=0,dev2=0;
 for(int i=0;i<6;i++){
   ee[i]=(double)e[i*n+z]-ep[i*n+z]-sqrt(1.5)*(double)u[i*n+z]*gamma[z];
   if(i<3)tr+=ee[i];
 }
 for(int i=0;i<6;i++){double v=ee[i]-(i<3?tr/3:0);dev2+=v*v;}
 const double kt=(double)k[z]+gamma[z],energy=mu*dev2+.5*K*tr*tr+.5*H*kt*kt;
 out[z]=(1-residual)*((double)d[z]-dn[z])*energy;
}
"""


_PARABOLOIDAL_WORK = r"""
extern "C" __global__ void pbd_damage_work(const REAL* e,const REAL* ep,const REAL* sig,const REAL* dp,
 const REAL* d,const REAL* dn,const int* ids,double* out,long long n,double mu,double K,
 double alpha,double rk,double residual){
 const long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=n)return;
 if(ids[z]){out[z]=0;return;}
 // The plastic increment is homogeneous of degree 0 in the (degraded) stress.
 REAL s[6],m;const REAL g=multiplier(sig,z,n,dp[z],(REAL)alpha,(REAL)rk,s,m);
 double ee[6],tr=0,dev2=0;
 for(int i=0;i<6;i++){
   ee[i]=(double)e[i*n+z]-ep[i*n+z]-(double)g*(3*s[i]+(i<3?2*alpha*m/3:0));
   if(i<3)tr+=ee[i];
 }
 for(int i=0;i<6;i++){double v=ee[i]-(i<3?tr/3:0);dev2+=v*v;}
 out[z]=(1-residual)*((double)d[z]-dn[z])*(mu*dev2+.5*K*tr*tr);
}
"""


class _ImplicitGradientDamage:
    """Confirmed/trial damage, nonlocal field and history shared by the laws.

    begin/propose/set_trial operate against a frozen confirmed DamageState.
    Only accept publishes a new state. Outer iterates may decrease, but never
    below confirmed damage; rejecting an attempt therefore cannot age damage.
    """

    def _setup_damage(self, parameters, lengths, timers, module):
        from cufhom.fft.nonlinear.nonlocal_damage import NonlocalHelmholtz

        cp = self.cp
        self.parameters = (
            parameters
            if isinstance(parameters, GradientDamageParameters)
            else GradientDamageParameters(**parameters)
        )
        p = self.parameters
        self.confirmed = DamageState(
            cp.zeros(self.shape, dtype=self.dtype),
            cp.zeros(self.shape, dtype=cp.float64),
        )
        self.trial = self.confirmed.value
        self.degradation = cp.ones(self.shape, dtype=self.dtype)
        # The heterogeneous diffusion action loses relative accuracy when
        # ell/h grows. Keep this scalar PDE in FP64, including its source;
        # mechanical fields/compact tangents retain the requested precision.
        self.source_buffer = cp.empty(self.shape, dtype=cp.float64)
        self.regularizer = NonlocalHelmholtz(
            self.shape,
            lengths,
            p.length_scale,
            self.ids == 0,
            xp=cp,
            dtype="float64",
            timers=timers,
        )
        self.interphase = (
            interphase_mask(cp, self.ids, lengths, p.interphase_thickness)
            if p.interphase_thickness
            else None
        )
        self.last_nonlocal = None
        self.candidate_history = self.confirmed.history
        self.source_kernel = module.get_function("gd_source")
        self.history_kernel = module.get_function("gd_history")
        self.degradation_kernel = cp.ElementwiseKernel(
            "T value, float64 residual",
            "T result",
            "result = (T)(residual + (1-residual)*(1-(double)value));",
            "gradient_damage_degradation",
        )

    @property
    def launch(self):
        return ((self.n + 255) // 256,), (256,)

    def begin(self):
        self.set_trial(self.confirmed.value)
        self.candidate_history = self.confirmed.history
        self.last_nonlocal = None

    def set_trial(self, value):
        self.trial = value
        self.degradation_kernel(
            value, np.float64(self.parameters.residual_stiffness), self.degradation
        )

    def source(self, strain):
        self.source_kernel(
            *self.launch,
            (
                strain,
                self.ids,
                self.source_buffer,
                np.int64(self.n),
                np.float64(self.material.nu),
                np.float64(self.parameters.strength_ratio),
            ),
        )
        return self.source_buffer

    def _thresholds(self):
        p = self.parameters
        if self.interphase is None:
            return np.uint64(0), np.float64(0), np.float64(1)
        return (
            self.interphase,
            np.float64(p.interphase_strain_threshold),
            np.float64(p.interphase_softening_strain),
        )

    def propose(self, strain):
        p = self.parameters
        field, diagnostics = self.regularizer.solve(
            self.source(strain),
            rtol=p.helmholtz_rtol,
            maxiter=p.helmholtz_max_iterations,
            initial=self.last_nonlocal,
        )
        h, d = (
            self.cp.empty(self.shape, dtype=self.cp.float64),
            self.cp.empty(self.shape, dtype=self.dtype),
        )
        self.history_kernel(
            *self.launch,
            (
                field,
                self.confirmed.history,
                self.confirmed.value,
                self.ids,
                h,
                d,
                np.int64(self.n),
                np.float64(p.strain_threshold),
                np.float64(p.softening_strain),
            )
            + self._thresholds(),
        )
        gap = float(self.cp.max(self.cp.abs(d - self.trial)))
        self.last_nonlocal, self.candidate_history = field, h
        return d, dict(diagnostics, damage_gap=gap)

    def accept(self):
        self.confirmed = DamageState(self.trial, self.candidate_history)

    def restore(self, value, history):
        cp = self.cp
        value, history = (
            cp.asarray(value, dtype=self.dtype),
            cp.asarray(history, dtype=cp.float64),
        )
        if value.shape != self.shape or history.shape != self.shape:
            raise ValueError("Invalid damage checkpoint shape")
        invalid = (
            (~cp.isfinite(value))
            | (~cp.isfinite(history))
            | (value < 0)
            | (value > 1)
            | (history < 0)
        )
        invalid |= (self.ids != 0) & ((value != 0) | (history != 0))
        if bool(cp.any(invalid)):
            raise ValueError("Invalid damage checkpoint values")
        p = self.parameters
        k0, ks = p.strain_threshold, p.softening_strain
        if self.interphase is not None:
            weak = self.interphase != 0
            k0 = cp.where(weak, p.interphase_strain_threshold, k0)
            ks = cp.where(weak, p.interphase_softening_strain, ks)
        safe = cp.maximum(history, k0)
        expected = cp.where(
            history > k0, 1 - k0 / safe * cp.exp(-(history - k0) / ks), 0.0
        )
        expected = cp.where(self.ids == 0, expected, 0.0)
        if (
            float(cp.max(cp.abs(value - expected)))
            > p.rtol + 32 * np.finfo(self.dtype).eps
        ):
            raise ValueError("Checkpoint damage and history mismatch")
        self.confirmed = DamageState(value, history)
        self.begin()


class CuPyGradientDamage(_ImplicitGradientDamage, dp.CuPyDruckerPrager):
    """Drucker-Prager matrix: fused return/degradation, compact tangent."""

    def __init__(
        self, cp, material, ids, table, dtype, parameters, lengths, timers=None
    ):
        super().__init__(cp, material, ids, table, dtype)
        # Reuse the complete cone/apex return. Only this variant reads g and
        # scales stress/five compact tangent maps in the same CUDA thread.
        code = dp._CUDA
        signature = " const int* ids,const REAL* table,REAL* sig"
        ending = "   sig[i*n+z]=v;\n }\n}"
        if code.count(signature) != 1 or code.count(ending) != 1:
            raise RuntimeError("Drucker-Prager kernel contract changed")
        code = code.replace(signature, " const REAL* gd," + signature.lstrip())
        code = code.replace(
            ending,
            ending[:-1]
            + """
 const REAL g=gd[z];aa[z]*=g;bb[z]*=g;bulk[z]*=g;cd[z]*=g;cp[z]*=g;
 for(int i=0;i<6;i++)sig[i*n+z]*=g;
}""",
        )
        mod = cp.RawModule(
            code=(code + _EXTRA)
            .replace("REAL", "float" if self.dtype == np.float32 else "double")
            .replace("MUL_RN", "__fmul_rn" if self.dtype == np.float32 else "__dmul_rn")
            .replace("ADD_RN", "__fadd_rn" if self.dtype == np.float32 else "__dadd_rn")
        )
        self.damage_update_kernel = mod.get_function("dp_update")
        self.damage_audit_kernel = mod.get_function("gd_audit")
        self.work_kernel = mod.get_function("gd_damage_work")
        self._setup_damage(parameters, lengths, timers, mod)

    def update(self, strain, state, start=None):
        """Evaluate a backward-Euler plastic trial response with fixed trial damage."""
        if start is not None:
            raise ValueError("Gradient damage uses backward_euler")
        cp, m, d = self.cp, self.material, self.dtype
        fields = [cp.empty(self.shape, dtype=d) for _ in range(6)]
        r = MaterialResponse(
            cp.empty_like(strain), cp.empty_like(strain), *fields[:3], *fields[3:]
        )
        self.damage_update_kernel(
            *self.launch,
            (
                strain,
                state.plastic_strain,
                state.equivalent,
                self.degradation,
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

    def audit(self, response, state):
        """Return trial yield-violation and plastic-work fields on the device."""
        violation = self.cp.empty(self.shape, dtype=self.dtype)
        work = self.cp.empty_like(violation)
        m = self.material
        self.damage_audit_kernel(
            *self.launch,
            (
                response.stress,
                response.dp,
                state.equivalent,
                self.degradation,
                self.ids,
                violation,
                work,
                np.int64(self.n),
                np.float64(m.yield_stress),
                np.float64(m.hardening),
                np.float64(m.friction),
                np.float64(m.beta),
            ),
        )
        return violation, work

    def damage_work(self, strain, state, response):
        """Return the device damage-work density for the supplied trial response."""
        out = self.cp.empty(self.shape, dtype=self.cp.float64)
        m = self.material
        self.work_kernel(
            *self.launch,
            (
                strain,
                state.plastic_strain,
                state.equivalent,
                response.direction,
                response.dp,
                self.trial,
                self.confirmed.value,
                self.ids,
                out,
                np.int64(self.n),
                np.float64(m.mu),
                np.float64(m.K),
                np.float64(m.hardening),
                np.float64(self.parameters.residual_stiffness),
            ),
        )
        return out


class CuPyParaboloidalDamage(_ImplicitGradientDamage, pb.CuPyParaboloidal):
    """Paraboloidal matrix with implicit-gradient damage (Melro-type plastic-damage).

    The return map is undamaged in effective stress; stress and the exact
    nonsymmetric compact tangent at fixed damage scale by the degradation.
    Yield audits use the effective stress. The damage driving energy is the
    elastic energy; hardening along the tabulated curves is dissipated.
    """

    def __init__(
        self, cp, material, ids, table, dtype, parameters, lengths, timers=None
    ):
        super().__init__(cp, material, ids, table, dtype)
        real = "float" if self.dtype == np.float32 else "double"
        identity = "#define MEM REAL\n#define PARAM REAL\n#define L(x) ((REAL)(x))\n#define S(x) (x)\n"
        code = (
            identity + pb._CUDA + pb._CUDA_REST + _EXTRA + _PARABOLOIDAL_WORK
        ).replace("REAL", real)
        code = code.replace("EPS", repr(float(np.finfo(self.dtype).eps)))
        mod = cp.RawModule(
            code=code.replace(
                "MUL_RN", "__fmul_rn" if real == "float" else "__dmul_rn"
            ).replace("ADD_RN", "__fadd_rn" if real == "float" else "__dadd_rn")
        )
        self.work_kernel = mod.get_function("pbd_damage_work")
        self._setup_damage(parameters, lengths, timers, mod)

    def _degradation(self):
        return self.degradation

    def damage_work(self, strain, state, response):
        """Return the device damage-work density for the supplied trial response."""
        out = self.cp.empty(self.shape, dtype=self.cp.float64)
        m = self.material
        self.work_kernel(
            *self.launch,
            (
                strain,
                state.plastic_strain,
                response.stress,
                response.dp,
                self.trial,
                self.confirmed.value,
                self.ids,
                out,
                np.int64(self.n),
                np.float64(m.mu),
                np.float64(m.K),
                np.float64(m.alpha),
                np.float64(m.equivalence),
                np.float64(self.parameters.residual_stiffness),
            ),
        )
        return out
