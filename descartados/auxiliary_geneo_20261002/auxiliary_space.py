"""Archived periodic Q1 auxiliary elasticity and local GenEO enrichment.

This changes only the PCG preconditioner. The fine spectral H, frequencies,
material coefficients and physical acceptance criterion remain unchanged.
"""
from contextlib import contextmanager
from itertools import product
import time
import os
from unittest.mock import patch
import numpy as np
from scipy import sparse, linalg
import cupy as cp
import cupyx.scipy.sparse as csparse
from cufhom.pipeline import linear_coordinates as lc
from cufhom_core import applications as app
from cufhom_core.tensors import fft

_CORNERS = np.array(list(product((0,1), repeat=3)))
_KERNELS = {}


def auxiliary_shape(shape, maximum=12):
    # Integer ratios permit exact adjoint gathering without float atomics.
    return tuple(max(d for d in range(2, min(n, maximum)+1) if n % d == 0)
                 if any(n % d == 0 for d in range(2, min(n, maximum)+1)) else n for n in shape)


def prolongation(fine, coarse):
    """Small CPU reference P, tensor product of periodic linear interpolation."""
    axes = []
    for n, m in zip(fine, coarse):
        positions = np.arange(n)*m/n
        left = np.floor(positions).astype(int); t = positions-left
        rows = np.repeat(np.arange(n), 2)
        cols = np.stack((left % m, (left+1) % m), axis=1).ravel()
        vals = np.stack((1-t, t), axis=1).ravel()
        axes.append(sparse.coo_matrix((vals,(rows,cols)), shape=(n,m)).tocsr())
    scalar = sparse.kron(sparse.kron(axes[0],axes[1]),axes[2],format='csr')
    return sparse.block_diag([scalar]*3, format='csr')


def element_matrices(coefficients, lengths, shape):
    """Eight-point Q1 quadrature, Mandel shear, full SPD tensor per cell."""
    h = np.asarray(lengths)/shape
    gauss = (np.array([-1.,1.])/np.sqrt(3.)+1)/2
    result = np.zeros((len(coefficients),24,24))
    for point in product(gauss,repeat=3):
        factors = np.where(_CORNERS, point, 1-np.asarray(point))
        gradients = np.empty((8,3))
        for axis in range(3):
            gradients[:,axis] = (2*_CORNERS[:,axis]-1)*np.prod(np.delete(factors,axis,axis=1),axis=1)/h[axis]
        b = np.zeros((6,24))
        for a, g in enumerate(gradients):
            for c in range(3): b[c,c*8+a] = g[c]
            b[3,1*8+a]=g[2]/np.sqrt(2); b[3,2*8+a]=g[1]/np.sqrt(2)
            b[4,0*8+a]=g[2]/np.sqrt(2); b[4,2*8+a]=g[0]/np.sqrt(2)
            b[5,0*8+a]=g[1]/np.sqrt(2); b[5,1*8+a]=g[0]/np.sqrt(2)
        result += np.einsum('ia,mij,jb->mab',b,coefficients,b)*(np.prod(h)/8)
    return result


def assemble(elements, shape):
    nodes = np.array(list(product(*(range(n) for n in shape))))
    corners = (nodes[:,None,:]+_CORNERS[None,:,:]) % np.asarray(shape)
    ids = np.ravel_multi_index(corners.transpose(2,0,1),shape)
    nd = int(np.prod(shape))
    dofs = np.concatenate([ids+c*nd for c in range(3)],axis=1)
    rows = np.broadcast_to(dofs[:,:,None],elements.shape).ravel()
    cols = np.broadcast_to(dofs[:,None,:],elements.shape).ravel()
    K = sparse.coo_matrix((elements.ravel(),(rows,cols)),shape=(3*nd,3*nd)).tocsr()
    return (K+K.T)*.5


def project(x, xp=np):
    parts = x.reshape(3,-1)
    return (parts-parts.mean(axis=1,keepdims=True)).ravel()


class Multigrid:
    """Fixed symmetric V-cycle on the mean-free auxiliary space."""
    def __init__(self, K, shape, xp=np):
        self.xp = xp; self.shape = tuple(shape)
        self.K = K if xp is np else csparse.csr_matrix(K.astype(np.float32))
        diagonal = K.diagonal()
        omega = .9/np.max(np.asarray(abs(K).sum(axis=1)).ravel()/diagonal)
        self.R = xp.asarray(omega/diagonal, dtype=xp.float64 if xp is np else xp.float32)
        self.P = self.child = self.inverse = None
        if max(shape) > 4 and all(n%2==0 for n in shape):
            coarse = tuple(n//2 for n in shape)
            P = prolongation(shape,coarse)
            self.child = Multigrid((P.T@K@P).tocsr(),coarse,xp)
            self.P = P if xp is np else csparse.csr_matrix(P.astype(np.float32))
        else:
            dense = K.toarray(); values,vectors = linalg.eigh(dense, check_finite=False)
            valid = values > values[-1]*1e-11
            self.inverse = xp.asarray((vectors[:,valid]/values[valid])@vectors[:,valid].T,
                                     dtype=xp.float64 if xp is np else xp.float32)

    def smooth(self, r):
        return project(self.R*project(r,self.xp),self.xp)

    def __call__(self,b):
        b = project(b,self.xp)
        if self.inverse is not None:
            return project(self.inverse@b,self.xp)
        x = self.smooth(b)
        x += self.P@self.child(self.P.T@(b-self.K@x))
        x += self.smooth(b-self.K@x)
        return project(x,self.xp)


def geneo_basis(K, elements, shape, threshold=.15):
    """KN v = lambda (D KD D) v; retain low-energy and rigid local modes.

    KN contains exactly the elements inside a periodic patch. KD is the
    principal restriction of global K. D is a multiplicity partition of unity.
    The threshold is an experimental Rayleigh quotient cutoff, not a claim of
    the GenEO paper's condition bound for the fine FFT operator.
    """
    shape = np.asarray(shape); nd = int(np.prod(shape))
    if min(shape)<5:raise ValueError('GenEO needs at least five periodic nodes per axis')
    cells = np.minimum(shape-2,4)
    starts = list(product(*(range(0,int(n),int(c)) for n,c in zip(shape,cells))))
    patches = []; multiplicity = np.zeros(nd)
    for start in starts:
        local_shape = cells+1
        points = np.array(list(product(*(range(int(n)) for n in local_shape))))
        global_nodes = np.ravel_multi_index(((points+start)%shape).T,shape)
        # A domain spanning the whole periodic axis would duplicate nodes.
        if len(np.unique(global_nodes)) != len(global_nodes):
            raise ValueError('GenEO patches must fit strictly inside the periodic domain')
        multiplicity[global_nodes] += 1
        patches.append((np.array(start),local_shape,global_nodes))
    columns = []; spectra=[]; residuals=[]
    for start,local_shape,nodes in patches:
        nlocal = len(nodes); dofs = np.concatenate([nodes+c*nd for c in range(3)])
        KN = np.zeros((3*nlocal,3*nlocal))
        for cell in product(*(range(int(c)) for c in cells)):
            corner = np.asarray(cell)+_CORNERS
            ids = np.ravel_multi_index(corner.T,local_shape)
            ld = np.concatenate([ids+c*nlocal for c in range(3)])
            cell_id = np.ravel_multi_index((start+cell)%shape,shape)
            KN[np.ix_(ld,ld)] += elements[cell_id]
        weight = np.tile(1/multiplicity[nodes],3)
        KD = K[dofs][:,dofs].toarray()
        metric = weight[:,None]*KD*weight[None,:]
        values,vectors = linalg.eigh(KN,metric,check_finite=False)
        choose = values <= threshold
        if values[0] < -1e-8*max(1.,values[-1]):
            raise ValueError('Indefinite local Neumann elasticity')
        spectra.append(dict(minimum=float(values[0]), maximum=float(values[-1]), selected=int(choose.sum()),
            selected_nonrigid=int(np.count_nonzero(choose & (values>1e-8))),
            smallest_positive=float(values[values>1e-8][0]),
            smallest_unselected=float(values[~choose][0]) if np.any(~choose) else None))
        for value,v in zip(values[choose],vectors[:,choose].T):
            denominator = np.linalg.norm(KN@v)+abs(value)*np.linalg.norm(metric@v)+1e-30
            # The absolute scaled check also covers exact rigid modes.
            residuals.append(float(np.linalg.norm(KN@v-value*(metric@v))/(np.linalg.norm(KN)*np.linalg.norm(v)+1e-30)))
            columns.append(sparse.coo_matrix((weight*v,(dofs,np.zeros(len(dofs),int))),shape=(3*nd,1)))
    Z = sparse.hstack(columns,format='csr')
    E = (Z.T@K@Z).toarray(); E = (E+E.T)*.5
    values,vectors = linalg.eigh(E,check_finite=False)
    valid = values > values[-1]*1e-10
    # Keep Z sparse: rank removal lives in the small inverse only. Translation
    # and dependent directions have zero inverse energy and are projected out.
    inverse = (vectors[:,valid]/values[valid])@vectors[:,valid].T
    return Z,inverse,dict(threshold=threshold,patches=len(patches),columns=Z.shape[1],
        rank=int(valid.sum()),local_spectra=spectra,max_scaled_eigen_residual=max(residuals),
        sparse_basis_bytes=int(Z.data.nbytes+Z.indices.nbytes+Z.indptr.nbytes))


_LIFT = r'''
extern "C" __global__ void lift(const float2* x,float2* y,
 long long n0,long long n1,long long n2,long long nz,
 float iy0,float iy1,float iy2,float norm,float gt,float gl,int forward) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 if(s==0||(n0%2==0&&i0==n0/2)||(n1%2==0&&i1==n1/2)||(n2%2==0&&i2==n2/2)) {
   for(int c=0;c<3;c++)y[c*count+s]=make_float2(0,0);return;
 }
 float k[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
 float kn=sqrtf(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
 for(int c=0;c<3;c++)k[c]/=kn;
 float weight=(i2==0||(n2%2==0&&i2==n2/2))?1:1.4142135623730950488f;
 float scale=6.2831853071795864769f*kn*norm*(forward?weight:1/weight);
 float re=0,im=0;
 for(int c=0;c<3;c++){re+=k[c]*x[c*count+s].x;im+=k[c]*x[c*count+s].y;}
 for(int c=0;c<3;c++)y[c*count+s]=make_float2(scale*(gt*x[c*count+s].x+(gl-gt)*k[c]*re),
                                                         scale*(gt*x[c*count+s].y+(gl-gt)*k[c]*im));
}
'''
_PROLONG = r'''
extern "C" __global__ void prolong(const float* x,float* y,
 long long n0,long long n1,long long n2,long long m0,long long m1,long long m2) {
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x,N=n0*n1*n2,M=m0*m1*m2;
 if(j>=3*N)return;
 int c=j/N;long long s=j%N,i[3]={s/(n1*n2),(s/n2)%n1,s%n2};
 long long m[3]={m0,m1,m2},n[3]={n0,n1,n2},left[3];float t[3];
 for(int a=0;a<3;a++){long long f=n[a]/m[a];left[a]=i[a]/f;t[a]=(float)(i[a]%f)/f;}
 float v=0;
 for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int d=0;d<2;d++) {
   long long k0=(left[0]+a)%m0,k1=(left[1]+b)%m1,k2=(left[2]+d)%m2;
   float w=(a?t[0]:1-t[0])*(b?t[1]:1-t[1])*(d?t[2]:1-t[2]);
   v+=w*x[c*M+(k0*m1+k1)*m2+k2];
 }y[j]=v;
}
'''
_RESTRICT = r'''
extern "C" __global__ void restrict_axis(const float* x,float* y,
 long long n0,long long n1,long long n2,long long m0,long long m1,long long m2,int axis) {
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x,M=m0*m1*m2,N=n0*n1*n2;
 if(j>=3*M)return;
 int c=j/M;long long s=j%M,i[3]={s/(m1*m2),(s/m2)%m1,s%m2};
 long long n[3]={n0,n1,n2},m[3]={m0,m1,m2},f=n[axis]/m[axis],center=i[axis]*f;
 float v=0;
 for(long long d=1-f;d<f;d++) {
   i[axis]=(center+d+n[axis])%n[axis];
   float w=1-fabsf((float)d)/f;
   v+=w*x[c*N+(i[0]*n1+i[1])*n2+i[2]];
 }y[j]=v;
}
'''


class Transfer:
    def __init__(self,Q,coarse):
        self.Q=Q; self.coarse=tuple(coarse)
        for name,code in [('lift',_LIFT),('prolong',_PROLONG),('restrict_axis',_RESTRICT)]:
            if name not in _KERNELS:_KERNELS[name]=cp.RawKernel(code,name)

    def fourier(self,x,forward):
        Q=self.Q; out=cp.empty((3,)+Q.hshape,cp.complex64)
        args=(x,out,*(np.int64(n) for n in Q.shape),np.int64(Q.hshape[-1]),
            *(np.float32(1/L) for L in Q.P.lengths),np.float32(Q.norm),
            np.float32(1/(np.sqrt(2)*Q.transverse)),np.float32(1/Q.longitudinal),np.int32(forward))
        _KERNELS['lift'](((Q.count+255)//256,),(256,),args)
        return out

    def prolong(self,x):
        out=cp.empty((3,)+self.Q.shape,cp.float32)
        _KERNELS['prolong'](((out.size+255)//256,),(256,),
            (x,out,*(np.int64(n) for n in self.Q.shape+self.coarse)))
        return out

    def restrict(self,x):
        shape=self.Q.shape
        for axis in range(3):
            target=list(shape);target[axis]=self.coarse[axis];target=tuple(target)
            out=cp.empty((3,)+target,cp.float32)
            _KERNELS['restrict_axis'](((out.size+255)//256,),(256,),
                (x,out,*(np.int64(n) for n in shape+target),np.int32(axis)))
            x=out;shape=target
        return x.ravel()

    def __call__(self,u):
        h=fft.rfftn_cached(self.prolong(u),self.Q.shape,(1,2,3))
        return self.fourier(h,True).ravel()

    def adjoint(self,r):
        h=self.fourier(r,False)
        # Raw inverse already has norm=1/sqrt(V) from the Fourier kernel.
        u=fft.irfftn_unscaled_cached(h,self.Q.shape,(1,2,3))
        return self.restrict(u)


def coarse_coefficients(descriptor,fine,coarse):
    indices=np.triu_indices(6);M=int(np.prod(coarse));values=np.empty((M,21))
    # Avoid a full fine C field: average one packed coefficient at a time.
    layout=tuple(v for pair in zip(coarse,np.asarray(fine)//coarse) for v in pair)
    for i in range(21):
        field=descriptor.material_table[:,i][descriptor.val].reshape(layout)
        values[:,i]=cp.asnumpy(field.mean(axis=(1,3,5),dtype=cp.float64)).ravel()
    C=np.zeros((M,6,6));C[:,indices[0],indices[1]]=values;C[:,indices[1],indices[0]]=values
    return C


class AuxiliaryPreconditioner:
    def __init__(self,Q,descriptor,geneo=False,fine_energy=False,action=None):
        started=time.perf_counter();cp.cuda.get_current_stream().synchronize()
        before=cp.get_default_memory_pool().used_bytes()
        self.transfer=Transfer(Q,auxiliary_shape(Q.shape))
        shape=self.transfer.coarse
        C=coarse_coefficients(descriptor,Q.shape,shape)
        # H uses voxel sums, while Q1 uses physical integrals.
        scale=np.prod(Q.shape)/np.prod(Q.P.lengths)
        elements=element_matrices(C,Q.P.lengths,shape)*scale
        K=assemble(elements,shape)
        self.S=None if fine_energy else Multigrid(K,shape,cp); self.Z=self.inverse=None
        self.info=dict(auxiliary_shape=list(shape),normalization='Q1 integral / fine voxel volume',
            type='GenEO auxiliary' if geneo else 'Q1 auxiliary V-cycle',fine_operator_unchanged=True)
        if geneo:
            Z,inverse,detail=geneo_basis(K,elements,shape,float(os.environ.get('CUFHOM_GENEO_THRESHOLD','.15')))
            self.Z=csparse.csr_matrix(Z.astype(np.float32)); self.inverse=cp.asarray(inverse,cp.float32)
            self.info['geneo']=detail
            if fine_energy:
                # U = T Z is represented by sparse auxiliary Z, never by
                # a bank of dense fine vectors. Use the actual fine H in E.
                energy=np.empty((Z.shape[1],Z.shape[1]))
                for column in range(Z.shape[1]):
                    z=cp.asarray(Z[:,column].toarray().ravel(),cp.float32)
                    hz=action(self.transfer(z))
                    energy[:,column]=cp.asnumpy(self.Z.T@self.transfer.adjoint(hz))
                del hz,z
                symmetry=float(np.linalg.norm(energy-energy.T)/np.linalg.norm(energy))
                if symmetry>2e-5:raise ValueError('Fine GenEO energy failed symmetry')
                values,vectors=linalg.eigh((energy+energy.T)*.5,check_finite=False)
                valid=values>values[-1]*1e-6
                inverse=(vectors[:,valid]/values[valid])@vectors[:,valid].T
                self.inverse=cp.asarray(inverse,cp.float32)
                self.info.update(type='GenEO with actual fine Galerkin energy',
                    fine_energy_setup_H=Z.shape[1],fine_energy_rank=int(valid.sum()),
                    fine_energy_symmetry=symmetry,fine_energy_min=float(values[valid][0]))
        cp.cuda.get_current_stream().synchronize()
        self.info.update(setup_s=time.perf_counter()-started,
            retained_gpu_bytes=cp.get_default_memory_pool().used_bytes()-before)

    def __call__(self,r):
        b=project(self.transfer.adjoint(r),cp)
        u=cp.zeros_like(b) if self.S is None else self.S(b)
        if self.Z is not None:
            u+=project(self.Z@(self.inverse@(self.Z.T@b)),cp)
        return r+self.transfer(u)


@contextmanager
def _context(geneo,fine_energy=False):
    original=lc.fused_cg;original_factory=lc.make_linear_solver;cached={};reported=set()
    original_payload=app._solver_info_payload
    def cg(action,rhs,dot,*args,**kw):
        captures=dict(zip(action.__code__.co_freevars,(c.cell_contents for c in action.__closure__)))
        Q,A=captures['Q'],captures['A']; descriptor=getattr(A,'op',A)
        key=(Q.shape,id(descriptor.val),id(descriptor.material_table))
        if key not in cached:cached[key]=AuxiliaryPreconditioner(Q,descriptor,geneo,fine_energy,action)
        precondition=cached[key]
        kw['precondition']=precondition
        # Metric smoothing is supplied by the separately tested PCG driver.
        from metric_pcg import metric_pcg
        return metric_pcg(action,rhs,dot,*args,**kw)
    def factory(**kwargs):
        solve=original_factory(**kwargs)
        def wrapped(*args,**kw):
            result,info=solve(*args,**kw)
            records=[]
            for key,precondition in cached.items():
                record=dict(precondition.info)
                if key in reported:record['setup_s']=0.
                else:reported.add(key)
                records.append(record)
            info['auxiliary_preconditioner']=records
            return result,info
        return wrapped
    def payload(info,load_ids):
        result=original_payload(info,load_ids)
        if 'auxiliary_preconditioner' in info:result['auxiliary_preconditioner']=info['auxiliary_preconditioner']
        return result
    with patch.object(lc,'fused_cg',cg),patch.object(lc,'make_linear_solver',factory),patch.object(app,'_solver_info_payload',payload):
        yield cached


def auxiliary_context():return _context(False)
def geneo_context():return _context(True)
def fine_geneo_context():return _context(True,True)
