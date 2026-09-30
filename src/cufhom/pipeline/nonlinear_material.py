"""Small-strain Mandel J2 with linear isotropic hardening (stress units: GPa).

The state is immutable during equilibrium iterations. A response contains only
the compact consistent tangent and the plastic increment needed for commit.
The CuPy kernels follow exactly the CPU formulas, one thread per voxel.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class J2Material:
    E: float = 3.0
    nu: float = 0.3
    yield_stress: float = 0.015
    hardening: float = 0.03

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.E, self.nu, self.yield_stress, self.hardening)):
            raise ValueError("Material parameters must be finite")
        if self.E <= 0 or not -1 < self.nu < .5 or self.yield_stress <= 0 or self.hardening <= 0:
            raise ValueError("Require E, yield_stress, hardening > 0 and -1 < nu < .5")

    @property
    def K(self):
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))


@dataclass
class PlasticState:
    plastic_strain: object
    equivalent: object


@dataclass
class MaterialResponse:
    stress: object
    direction: object
    a: object
    b: object
    dp: object


def initial_state(shape, xp=np, dtype=np.float64):
    return PlasticState(xp.zeros((6,) + tuple(shape), dtype=dtype), xp.zeros(shape, dtype=dtype))


def elastic_action(value, ids, table):
    """CPU indexed Mandel matrix multiplication, without a dense C(x)."""
    out = np.zeros_like(value)
    for i in range(6):
        for j in range(6):
            out[i] += table[ids, i, j] * value[j]
    return out


def update_cpu(strain, state, material, ids, table):
    mu, K = material.mu, material.K
    ee = strain - state.plastic_strain
    trace = ee[:3].sum(axis=0)
    trial = 2 * mu * ee
    trial[:3] -= (2 * mu / 3) * trace
    norm = np.sqrt(np.sum(trial * trial, axis=0))
    q = np.sqrt(1.5) * norm
    dp = np.maximum(q - material.yield_stress - material.hardening * state.equivalent, 0)
    dp /= 3 * mu + material.hardening
    dp = np.where(ids == 0, dp, 0)
    direction = trial / np.maximum(norm, np.finfo(strain.dtype).tiny)
    ratio = dp / np.maximum(q, np.finfo(strain.dtype).tiny)
    a = 2 * mu * (1 - 3 * mu * ratio)
    b = np.where(dp > 0, -6 * mu * mu * (1 / (3 * mu + material.hardening) - ratio), 0)
    stress = trial * (a / (2 * mu))
    stress[:3] += K * trace
    if np.any(ids):
        fiber_stress = elastic_action(strain, ids, table)
        stress[:, ids != 0] = fiber_stress[:, ids != 0]
    return MaterialResponse(stress, direction, a, b, dp)


def tangent_cpu(value, response, material, ids, table):
    trace = value[:3].sum(axis=0)
    out = response.a * value + response.b * response.direction * np.sum(response.direction * value, axis=0)
    out[:3] += (material.K - response.a / 3) * trace
    if np.any(ids):
        fiber = elastic_action(value, ids, table)
        out[:, ids != 0] = fiber[:, ids != 0]
    return out


def commit_state(state, response):
    """Allocate a new state; neither the response nor the old state is mutated."""
    return PlasticState(state.plastic_strain + np.sqrt(1.5) * response.direction * response.dp,
                        state.equivalent + response.dp)


_CUDA = r'''
extern "C" __global__ void update(const REAL* e, const REAL* ep, const REAL* alpha,
 const int* ids, const REAL* table, REAL* sig, REAL* u, REAL* a, REAL* b, REAL* dp,
 long long n, REAL mu, REAL K, REAL sy, REAL H) {
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return;
 REAL s[6], tr=0, norm2=0;
 for(int i=0;i<3;i++)tr+=e[i*n+k]-ep[i*n+k];
 for(int i=0;i<6;i++){s[i]=2*mu*(e[i*n+k]-ep[i*n+k]-(i<3?tr/3:0)); norm2+=s[i]*s[i];}
 REAL norm=sqrt(norm2), q=sqrt((REAL)1.5)*norm;
 REAL d=ids[k]==0?fmax((REAL)0,(q-sy-H*alpha[k])/(3*mu+H)):(REAL)0;
 REAL ratio=q>0?d/q:0; dp[k]=d; a[k]=2*mu*(1-3*mu*ratio);
 b[k]=d>0?-6*mu*mu*(1/(3*mu+H)-ratio):0;
 for(int i=0;i<6;i++){
   u[i*n+k]=norm>0?s[i]/norm:0;
   REAL v=s[i]*(a[k]/(2*mu))+(i<3?K*tr:0);
   if(ids[k]){v=0; for(int j=0;j<6;j++)v+=table[ids[k]*36+i*6+j]*e[j*n+k];}
   sig[i*n+k]=v;
 }
}
extern "C" __global__ void tangent(const REAL* v, const REAL* u, const REAL* a,
 const REAL* b, const int* ids, const REAL* table, REAL* out, long long n, REAL K){
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(k>=n)return;
 REAL tr=v[k]+v[n+k]+v[2*n+k], dot=0;
 for(int i=0;i<6;i++)dot+=u[i*n+k]*v[i*n+k];
 for(int i=0;i<6;i++){
   REAL w=a[k]*v[i*n+k]+b[k]*u[i*n+k]*dot+(i<3?(K-a[k]/3)*tr:0);
   if(ids[k]){w=0;for(int j=0;j<6;j++)w+=table[ids[k]*36+i*6+j]*v[j*n+k];}
   out[i*n+k]=w;
 }
}
'''


class CuPyMaterial:
    def __init__(self, cp, material, ids, table, dtype):
        self.cp, self.material, self.ids, self.table = cp, material, ids, table
        self.dtype = np.dtype(dtype).type
        self.shape, self.n = ids.shape, ids.size
        mod = cp.RawModule(code=_CUDA.replace('REAL', 'float' if self.dtype == np.float32 else 'double'))
        self.update_kernel, self.tangent_kernel = mod.get_function('update'), mod.get_function('tangent')

    def update(self, strain, state):
        cp, m, d = self.cp, self.material, self.dtype
        response = MaterialResponse(cp.empty_like(strain), cp.empty_like(strain),
                                    cp.empty(self.shape, dtype=d), cp.empty(self.shape, dtype=d),
                                    cp.empty(self.shape, dtype=d))
        self.update_kernel(((self.n+255)//256,), (256,),
                           (strain, state.plastic_strain, state.equivalent, self.ids, self.table,
                            response.stress, response.direction, response.a, response.b, response.dp,
                            np.int64(self.n), d(m.mu), d(m.K), d(m.yield_stress), d(m.hardening)))
        return response

    def tangent(self, value, response):
        out = self.cp.empty_like(value)
        self.tangent_kernel(((self.n+255)//256,), (256,),
                            (value, response.direction, response.a, response.b, self.ids, self.table,
                             out, np.int64(self.n), self.dtype(self.material.K)))
        return out
