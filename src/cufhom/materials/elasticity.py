"""Elastic stiffness, Mandel conversions and CPU/GPU material rotations (11,22,33,23,13,12)."""
from typing import Tuple
import numpy as np
from cufhom_core.tensors.fft import cp

_KERNELS={}
_SOURCE=r'''
__device__ double bond(const double* R,int row,int column) {
 const int p[6]={0,1,2,1,0,0},q[6]={0,1,2,2,2,1};
 int i=p[row],j=q[row],a=p[column],b=q[column];
 double factor=row<3?1.:1.4142135623730950488;
 double value=R[3*i+a]*R[3*j+b];
 if(column>=3)value=(value+R[3*i+b]*R[3*j+a])*.7071067811865475244;
 return factor*value;
}
extern "C" __global__ void rotate_mandel(const double* R,const double* C,REAL* output,
 long long groups,int width) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(s>=groups*width)return;
 long long group=s/width;int index=s%width,i,j;
 if(width==36){i=index/6;j=index%6;}
 else {i=0;int first=0;while(index>=first+6-i){first+=6-i;i++;}j=i+index-first;}
 double mi[6],mj[6];
 #pragma unroll
 for(int a=0;a<6;a++){mi[a]=bond(R+9*group,i,a);mj[a]=bond(R+9*group,j,a);}
 double sum=0;
 #pragma unroll
 for(int a=0;a<6;a++) {
   double inner=0;
   #pragma unroll
   for(int b=0;b<6;b++)inner+=C[6*a+b]*mj[b];
   sum+=mi[a]*inner;
 }
 output[s]=(REAL)sum;
}
'''


def rotate_mandel_batch_gpu(C,rotations,dtype,packed_sym21,chunk_size=0):
    """Return one rotated 6×6 or upper-triangle table per supplied rotation.

    The caller supplies orthogonal 3×3 rotations and a symmetric 6×6 matrix,
    already validated by the material builder. Chunking bounds rotation
    temporaries without changing group order or coefficient packing.
    """
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Mandel table requires float32 or float64')
    if dtype.str not in _KERNELS:
        _KERNELS[dtype.str]=cp.RawKernel(
            _SOURCE.replace('REAL','float' if dtype==np.float32 else 'double'),'rotate_mandel')
    width=21 if packed_sym21 else 36
    result=cp.empty((len(rotations),width),dtype=dtype)
    if not len(rotations):return result if packed_sym21 else result.reshape((0,6,6))
    tensor=cp.ascontiguousarray(cp.asarray(C,dtype=cp.float64))
    chunk=int(chunk_size) if chunk_size else len(rotations)
    chunk=max(1,min(chunk,len(rotations)))
    for first in range(0,len(rotations),chunk):
        rotation=cp.ascontiguousarray(cp.asarray(rotations[first:first+chunk],dtype=cp.float64))
        n=len(rotation)
        _KERNELS[dtype.str](((n*width+127)//128,),(128,),
            (rotation,tensor,result[first:first+n],np.int64(n),np.int32(width)))
    return result if packed_sym21 else result.reshape((len(rotations),6,6))


def mandel_to_voigt_eng(Cm):
    """
    Mandel (6x6) -> Voigt ingenieril (6x6).
    Voigt ingenieril usa gamma_ij = 2*epsilon_ij, y C44=G.
    """
    T_inv = np.diag([1, 1, 1, 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
    return T_inv @ Cm @ T_inv

def engineering_constants_from_Cmandel(Cm):
    Cm = 0.5*(Cm + Cm.T)

    Cv = mandel_to_voigt_eng(Cm)
    S  = np.linalg.inv(Cv)

    E1 = 1.0 / S[0,0]
    E2 = 1.0 / S[1,1]
    E3 = 1.0 / S[2,2]

    # Voigt ingenieril: 4->23, 5->13, 6->12
    G23 = 1.0 / S[3,3]
    G13 = 1.0 / S[4,4]
    G12 = 1.0 / S[5,5]

    nu12 = -S[0,1] / S[0,0]
    nu13 = -S[0,2] / S[0,0]
    nu23 = -S[1,2] / S[1,1]

    return dict(E1=E1,E2=E2,E3=E3, G12=G12,G13=G13,G23=G23, nu12=nu12,nu13=nu13,nu23=nu23)


def Enu_to_KG(E: float, nu: float) -> Tuple[float, float]:
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return K, G


def TI_stiffness_voigt(E_L: float, E_T: float, nu_LT: float, nu_TT: float, G_LT: float) -> np.ndarray:
    G_TT = E_T / (2.0 * (1.0 + nu_TT))
    S = np.zeros((6, 6), dtype=float)
    S[0, 0] = 1.0 / E_L;  S[1, 1] = 1.0 / E_T;  S[2, 2] = 1.0 / E_T
    S[0, 1] = S[1, 0] = -nu_LT / E_L
    S[0, 2] = S[2, 0] = -nu_LT / E_L
    S[1, 2] = S[2, 1] = -nu_TT / E_T
    S[3, 3] = 1.0 / G_TT;  S[4, 4] = 1.0 / G_LT;  S[5, 5] = 1.0 / G_LT
    return np.linalg.inv(S)


def voigt_to_mandel(Cv: np.ndarray) -> np.ndarray:
    T = np.diag([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    return T @ Cv @ T


def mandel_to_tensor4(Cm: np.ndarray) -> np.ndarray:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    fac = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)], dtype=float)
    C4 = np.zeros((3, 3, 3, 3), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            val = Cm[I, J] / (fac[I] * fac[J])
            C4[i, j, k, l] = val;  C4[j, i, k, l] = val
            C4[i, j, l, k] = val;  C4[j, i, l, k] = val
    return C4


def tensor4_to_mandel(C4: np.ndarray) -> np.ndarray:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    fac = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)], dtype=float)
    Cm = np.zeros((6, 6), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            Cm[I, J] = fac[I] * fac[J] * C4[i, j, k, l]
    return Cm


def rotation_matrix_from_vector(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    e1 = a / n
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, e1)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(e1, tmp)
    e2n = np.linalg.norm(e2)
    if e2n < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0])
        e2 = np.cross(e1, tmp)
        e2n = np.linalg.norm(e2)
        if e2n < 1e-12:
            return np.eye(3)
    e2 /= e2n
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def rotate_C_mandel(Cm_local: np.ndarray, R: np.ndarray) -> np.ndarray:
    C4 = mandel_to_tensor4(Cm_local)
    C4g = np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, C4)
    return tensor4_to_mandel(C4g)

