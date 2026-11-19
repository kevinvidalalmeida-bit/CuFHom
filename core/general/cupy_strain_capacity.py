"""Resident strain CG: share a padded GPU buffer across the 3-D library FFTs.

Only the private indexed/direct-projection workflow opts in. Padded material
output has the same FP32/FP64 arithmetic as the ordinary indexed operator. cuFFT
operates on aliased real and complex views; a component-sized GPU scratch
compacts the final field without transferring the active iterate to host.
"""
import numpy as np
import cupy as cp
from cufhom_core.tensors.objects import _get_cupy_indexed_sym21_kernel
_PLANS={}
_KERNELS={}


def supported(operator,rhs):
    from cufhom_core.tensors.operators import Operator,DFT,Tensor,_unwrap_timed
    from cufhom_core.general.cupy_capacity import _device_capacity
    if (type(operator) is not Operator or not isinstance(rhs.val,cp.ndarray)
            or rhs.val.dtype not in (cp.float32,cp.float64) or not rhs.val.flags.c_contiguous
            or rhs.Fourier or rhs.val.ndim!=4 or rhs.val.shape[0]!=6
            or len(operator.mat_rev)!=1 or len(operator.mat_rev[0])!=2):
        return False
    material,projection=operator.mat_rev[0]
    if type(projection) is not Operator or len(projection.mat_rev)!=1 or len(projection.mat_rev[0])!=3:
        return False
    forward,direct,inverse=projection.mat_rev[0]
    if (not all(_unwrap_timed(x) is x for x in [material,forward,direct,inverse])
            or not isinstance(material,Tensor) or material.multype not in ('sym21_indexed','indexed_sym21')
            or not isinstance(direct,Tensor) or direct.multype not in ('elasticity_hg1_direct','elasticity_g1_direct')
            or type(forward) is not DFT or type(inverse) is not DFT
            or forward.inverse or not inverse.inverse or forward.fft_form!='r' or inverse.fft_form!='r'
            or tuple(forward.N)!=tuple(rhs.N) or tuple(inverse.N)!=tuple(rhs.N)
            or not isinstance(material.val,cp.ndarray) or material.val.dtype!=cp.int32
            or not material.val.flags.c_contiguous or tuple(material.N)!=tuple(rhs.N)
            or tuple(material.val.shape) not in (tuple(rhs.N),(1,)+tuple(rhs.N))
            or not isinstance(getattr(material,'material_table',None),cp.ndarray)
            or material.material_table.dtype!=rhs.val.dtype or not material.material_table.flags.c_contiguous
            or material.material_table.ndim!=2 or material.material_table.shape[1]!=21
            or material.material_table.device.id!=rhs.val.device.id
            or material.val.device.id!=rhs.val.device.id):
        return False
    # Three resident CG vectors, one padded transform field, one component of
    # compaction scratch, material indices and extra headroom for descriptors.
    return (27*rhs.val.dtype.itemsize+4)*int(np.prod(rhs.N)) < _device_capacity(cp.cuda.runtime.getDevice())-512*2**20


def plans(shape,dtype=np.float32):
    shape=tuple(shape)
    dtype=np.dtype(dtype)
    key=(cp.cuda.runtime.getDevice(),shape,dtype.str)
    if key not in _PLANS:
        half=shape[:-1]+(shape[-1]//2+1,)
        padded=shape[:-1]+(2*half[-1],)
        forward=cp.cuda.cufft.CUFFT_R2C if dtype==np.dtype('float32') else cp.cuda.cufft.CUFFT_D2Z
        inverse=cp.cuda.cufft.CUFFT_C2R if dtype==np.dtype('float32') else cp.cuda.cufft.CUFFT_Z2D
        with cp.cuda.using_allocator():
            f=cp.cuda.cufft.PlanNd(shape,padded,1,int(np.prod(padded)),half,1,int(np.prod(half)),forward,1,'C',2,half[-1])
            b=cp.cuda.cufft.PlanNd(shape,half,1,int(np.prod(half)),padded,1,int(np.prod(padded)),inverse,1,'C',2,shape[-1])
        _PLANS[key]=(f,b)
    return _PLANS[key]

def kernel(dtype=np.float32):
    key=np.dtype(dtype).str
    if key not in _KERNELS:
        code=_get_cupy_indexed_sym21_kernel(dtype).code
        code=code.replace('void indexed_sym21_apply(','void indexed_sym21_padded(')
        code=code.replace('const long materials) {','const long materials, const long last, const long outlast, const long outspatial) {')
        code=code.replace('const long stride = batch * spatial;','const long stride = batch * spatial;\n const long obase = b * outspatial + (s / last) * outlast + s % last;\n const long ostride = batch * outspatial;')
        code=code.replace('Y[c*stride+base]','Y[c*ostride+obase]')
        for c in range(6):
            code=code.replace(f'Y[{c} * stride + base]',f'Y[{c} * ostride + obase]')
        _KERNELS[key]=cp.RawKernel(code,'indexed_sym21_padded')
    return _KERNELS[key]

def apply(material,projection,value,shape):
    from cufhom_core.tensors.operators import _tensor_apply_cupy
    shape=tuple(int(n) for n in shape)
    half=shape[:-1]+(shape[-1]//2+1,)
    padded=shape[:-1]+(2*half[-1],)
    f,b=plans(shape,value.dtype)
    work=cp.empty((6,)+padded,dtype=value.dtype)
    spectrum=work.view(cp.complex64 if value.dtype==cp.float32 else cp.complex128)
    real=work[...,:shape[-1]]
    N=int(np.prod(shape))
    NP=int(np.prod(padded))
    kernel(value.dtype)(((N+255)//256,),(256,),(material.val,material.material_table,value,work,np.int64(1),np.int64(N),np.int64(N),np.int64(material.material_table.shape[0]),np.int64(shape[-1]),np.int64(padded[-1]),np.int64(NP)))
    for c in range(6):
        f.fft(real[c],spectrum[c],cp.cuda.cufft.CUFFT_FORWARD)
    # Preserve the existing direct-projector/raw-C2R operator convention. The
    # analytic projector does not read its placeholder coefficient array.
    _tensor_apply_cupy(projection,spectrum,out=spectrum)
    for c in range(6):
        b.fft(spectrum[c],real[c],cp.cuda.cufft.CUFFT_INVERSE)
    # Sequential component compaction. A component-sized staging buffer protects
    # each source; earlier destinations never reach later component sources.
    scratch=cp.empty(shape,dtype=value.dtype)
    out=cp.ndarray((6,)+shape,dtype=value.dtype,memptr=work.data)
    for c in range(6):
        cp.copyto(scratch,real[c])
        cp.copyto(out[c],scratch)
    return out
