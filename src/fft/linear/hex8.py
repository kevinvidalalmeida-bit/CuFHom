"""Periodic, fully integrated HEX8 benchmark with a nodal true-residual audit.

Arrays use Z,Y,X spatial order and X,Y,Z displacement components. Strains use
Mandel order 11,22,33,12,13,23. This separate benchmark does not select or modify
the production Fourier Galerkin solver. Element energy is normalized by the
domain volume, as in the published unit-cube FANS problem.
"""
from itertools import product

import numpy as np
from cufhom.fft.krylov import _cg_updates
from cufhom_core.tensors.fft import cp


def element_operators(shape, lengths):
    """Eight Gauss points and trilinear shape-function derivatives."""
    nodes = np.array([(n >> 2 & 1, n >> 1 & 1, n & 1) for n in range(8)])
    h = np.asarray(lengths, float) / shape
    points = (.5 - .5 / np.sqrt(3), .5 + .5 / np.sqrt(3))
    B = np.zeros((8, 6, 24))
    for g, point in enumerate(product(points, repeat=3)):
        for a, node in enumerate(nodes):
            factors = np.where(node, point, 1 - np.asarray(point))
            gradient = [(2 * node[d] - 1) * np.prod(np.delete(factors, d)) / h[d]
                        for d in range(3)]
            z, y, x = gradient
            B[g, :3, 3*a:3*a+3] = np.diag([x, y, z])
            B[g, 3, 3*a:3*a+3] = [y, x, 0] / np.sqrt(2)
            B[g, 4, 3*a:3*a+3] = [z, 0, x] / np.sqrt(2)
            B[g, 5, 3*a:3*a+3] = [0, z, y] / np.sqrt(2)
    return nodes, B


_ACTION = r'''
extern "C" __global__ void hex8_action(const unsigned char* pattern,
 const double* table, const double* u, double* out, int nz,int ny,int nx) {
 long n=(long)nz*ny*nx, i=(long)blockDim.x*blockIdx.x+threadIdx.x;
 if(i>=n) return;
 int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny),p=pattern[i],o=0;
 double a=0,b=0,c=0;
 for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++)
 for(int dx=-1;dx<=1;dx++,o++) {
   int xx=(x+dx+nx)%nx,yy=(y+dy+ny)%ny,zz=(z+dz+nz)%nz;
   long j=((long)zz*ny+yy)*nx+xx;
   double v0=u[j],v1=u[n+j],v2=u[2*n+j];
   const double* t=table+o*9*256+p;
   a+=t[0*256]*v0+t[1*256]*v1+t[2*256]*v2;
   b+=t[3*256]*v0+t[4*256]*v1+t[5*256]*v2;
   c+=t[6*256]*v0+t[7*256]*v1+t[8*256]*v2;
 }
 out[i]=a;out[n+i]=b;out[2*n+i]=c;
}
'''

_SYMBOL = r'''
extern "C" __global__ void hex8_inverse(const double* t,double* w,
 int nz,int ny,int nx) {
 int nh=nx/2+1;long n=(long)nz*ny*nh,i=(long)blockDim.x*blockIdx.x+threadIdx.x;
 if(i>=n)return;
 if(i==0){for(int j=0;j<6;j++)w[j*n]=0;return;}
 double ax=6.2831853071795864769*(i%nh)/nx;
 double ay=6.2831853071795864769*((i/nh)%ny)/ny;
 double az=6.2831853071795864769*(i/(nh*ny))/nz;
 double cx=cos(ax),sx=sin(ax),cy=cos(ay),sy=sin(ay),cz=cos(az),sz=sin(az);
 double a=0,b=0,c=0,d=0,e=0,f=0;int o=0;
 for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++,o++){
   double rx=dx?cx:1,ry=dy?cy:1,rz=dz?cz:1;
   double ix=dx*sx,iy=dy*sy,iz=dz*sz;
   double q=rx*ry*rz-ix*iy*rz-ix*ry*iz-rx*iy*iz;
   a+=t[o*9+0]*q;b+=t[o*9+4]*q;c+=t[o*9+8]*q;
   d+=t[o*9+1]*q;e+=t[o*9+2]*q;f+=t[o*9+5]*q;
 }
 double den=a*b*c+2*d*e*f-a*f*f-b*e*e-c*d*d;
 w[i]=(b*c-f*f)/den;w[n+i]=(a*c-e*e)/den;w[2*n+i]=(a*b-d*d)/den;
 w[3*n+i]=(e*f-d*c)/den;w[4*n+i]=(d*f-e*b)/den;w[5*n+i]=(d*e-a*f)/den;
}
'''

_PRECONDITION = r'''
extern "C" __global__ void hex8_precondition(const double* w,const double2* r,
 double2* out,long n){
 long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double2 x=r[i],y=r[n+i],z=r[2*n+i];
 double a=w[i],b=w[n+i],c=w[2*n+i],d=w[3*n+i],e=w[4*n+i],f=w[5*n+i];
 out[i]=make_double2(a*x.x+d*y.x+e*z.x,a*x.y+d*y.y+e*z.y);
 out[n+i]=make_double2(d*x.x+b*y.x+f*z.x,d*x.y+b*y.y+f*z.y);
 out[2*n+i]=make_double2(e*x.x+f*y.x+c*z.x,e*x.y+f*y.y+c*z.y);
}
'''


class PeriodicHex8:
    """Binary isotropic HEX8 operator; translations are the only removed modes.

    The 256 incident-phase patterns compress the assembled nodal stencil
    exactly. No stiffness entries, quadrature points or Nyquist modes are
    discarded. The FFT inverse uses the arithmetic mean of phase Lamé moduli.
    """
    def __init__(self, phase, E=(75., 400.), nu=(.3, .2), lengths=(1., 1., 1.), xp=np):
        if xp is not np and (cp is None or xp is not cp):
            raise ValueError('HEX8 backend must be NumPy or available CuPy')
        phase = np.asarray(phase)
        if phase.ndim != 3 or min(phase.shape) < 2 or not np.isin(phase, [0, 1]).all():
            raise ValueError('HEX8 requires a binary three-dimensional grid, at least two cells per axis')
        E, nu, lengths = np.asarray(E, float), np.asarray(nu, float), np.asarray(lengths, float)
        if (E.shape != (2,) or nu.shape != (2,) or lengths.shape != (3,)
                or not np.isfinite(np.r_[E, nu, lengths]).all()
                or (E <= 0).any() or (lengths <= 0).any()
                or (nu <= -1).any() or (nu >= .5).any()):
            raise ValueError('Invalid isotropic phase constants or domain lengths')
        self.shape, self.xp = phase.shape, xp
        self.nodes, self.B = element_operators(np.array(self.shape), lengths)
        lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
        self.C = 2*mu[:, None, None]*np.eye(6)[None]
        self.C[:, :3, :3] += lam[:, None, None]
        self.Ke = np.einsum('gai,pab,gbj->pij', self.B, self.C, self.B) / (8*phase.size)
        patterns = np.zeros(self.shape, np.uint8)
        for a, node in enumerate(self.nodes):
            patterns |= np.roll(phase.astype(np.uint8), tuple(node), (0, 1, 2)) << a
        offsets = list(product((-1, 0, 1), repeat=3))
        offset_ids = {offset: i for i, offset in enumerate(offsets)}
        self.offsets = offsets
        self.pattern_phases = ((np.arange(256)[:, None] >> np.arange(8)) & 1)
        table = np.zeros((256, 27, 3, 3))
        for a, na in enumerate(self.nodes):
            for b, nb in enumerate(self.nodes):
                o = offset_ids[tuple(nb-na)]
                table[:, o] += self.Ke[self.pattern_phases[:, a], 3*a:3*a+3, 3*b:3*b+3]
        self.pattern = xp.asarray(patterns)
        self.phase = xp.asarray(phase.astype(np.uint8))
        self.table = xp.asarray(table.transpose(1, 2, 3, 0).copy())
        homogeneous = (table[0]+table[-1])*.5
        np.testing.assert_allclose(homogeneous, homogeneous.transpose(0, 2, 1), atol=1e-12)
        if xp is np:
            frequencies = [np.fft.fftfreq(n)*2*np.pi for n in self.shape]
            frequencies[-1] = np.fft.rfftfreq(self.shape[-1])*2*np.pi
            kz, ky, kx = np.meshgrid(*frequencies, indexing='ij')
            symbol = sum(block*np.cos(dz*kz+dy*ky+dx*kx)[..., None, None]
                         for (dz, dy, dx), block in zip(offsets, homogeneous))
            symbol[0, 0, 0] = np.eye(3)
            self.inverse = np.linalg.inv(symbol)
            self.inverse[0, 0, 0] = 0
        else:
            self.action_kernel = cp.RawKernel(_ACTION, 'hex8_action')
            self.prec_kernel = cp.RawKernel(_PRECONDITION, 'hex8_precondition')
            self.inverse = cp.empty((6, *self.shape[:2], self.shape[-1]//2+1), dtype=cp.float64)
            n = self.inverse[0].size
            cp.RawKernel(_SYMBOL, 'hex8_inverse')(((n+255)//256,), (256,),
                (cp.asarray(homogeneous), self.inverse, *map(np.int32, self.shape)))
            if not bool(cp.isfinite(self.inverse).all()):
                raise RuntimeError('Nonfinite HEX8 reference inverse')

    def action(self, u):
        xp = self.xp
        u = xp.ascontiguousarray(u, dtype=xp.float64)
        if u.shape != (3, *self.shape):
            raise ValueError('Expected X,Y,Z nodal displacement components on a Z,Y,X grid')
        out = xp.zeros_like(u)
        if xp is np:
            for o, delta in enumerate(self.offsets):
                values = np.roll(u, tuple(-d for d in delta), (1, 2, 3))
                block = np.take(self.table[o], self.pattern, axis=-1)
                out += np.einsum('ij...,j...->i...', block, values)
        else:
            n = self.phase.size
            self.action_kernel(((n+255)//256,), (256,),
                (self.pattern, self.table, u, out, *map(np.int32, self.shape)))
        return out

    def rhs(self, macro):
        macro = np.asarray(macro, float)
        if macro.shape != (6,) or not np.isfinite(macro).all():
            raise ValueError('Expected six finite Mandel components in 11,22,33,12,13,23 order')
        force = -np.einsum('gai,pab,b->pi', self.B, self.C, macro)/(8*self.phase.size)
        force = force.reshape(2, 8, 3)
        by_pattern = sum(force[self.pattern_phases[:, a], a] for a in range(8))
        # Constant affine stress has exactly zero nodal force. Preserve that
        # cancellation explicitly instead of solving a roundoff-only RHS.
        by_pattern[[0, 255]] = 0
        if np.array_equal(self.C[0], self.C[1]):
            by_pattern[:] = 0
        return self.xp.ascontiguousarray(self.xp.asarray(by_pattern)[self.pattern].transpose(3, 0, 1, 2))

    def precondition(self, residual):
        xp = self.xp
        spectrum = xp.fft.rfftn(residual, axes=(1, 2, 3))
        if xp is np:
            value = np.einsum('...ij,j...->i...', self.inverse, spectrum)
        else:
            value = cp.empty_like(spectrum)
            n = spectrum[0].size
            self.prec_kernel(((n+255)//256,), (256,), (self.inverse, spectrum, value, np.int64(n)))
        return xp.fft.irfftn(value, s=self.shape, axes=(1, 2, 3))

    def stress_field(self, u, macro):
        """Cell stress averaged over all eight Gauss points, on the CPU."""
        if self.xp is not np:
            u, phase = cp.asnumpy(u), cp.asnumpy(self.phase)
        else:
            phase = self.phase
        strain = np.broadcast_to(np.asarray(macro)[:, None, None, None], (6, *self.shape)).copy()
        Bbar = self.B.mean(0)
        for a, node in enumerate(self.nodes):
            values = np.roll(u, tuple(-int(d) for d in node), (1, 2, 3))
            strain += np.einsum('ij,j...->i...', Bbar[:, 3*a:3*a+3], values)
        stress = np.empty_like(strain)
        for p in (0, 1):
            mask = phase == p
            stress[:, mask] = self.C[p]@strain[:, mask]
        return stress

    def mean_stress(self, u, macro):
        """Mean of the full-integration cell stresses, in GPa."""
        return self.stress_field(u, macro).mean(axis=(1, 2, 3))


def solve_hex8(operator, macro, rtol=1e-6, maxiter=2000):
    """PCG with infinity-norm acceptance of freshly evaluated b-Ku.

    Includes RHS and the final audit. Reuses CuFHom's fused vector updates on
    GPU. The FFT preconditioner retains all nonzero HEX8 frequencies.
    """
    if not np.isfinite(rtol) or rtol <= 0 or maxiter < 1:
        raise ValueError('Positive tolerance and iteration limit required')
    xp = operator.xp
    rhs = operator.rhs(macro)
    threshold = rtol*float(xp.abs(rhs).max())
    x, r = xp.zeros_like(rhs), rhs.copy()
    if threshold == 0:
        return dict(solution=x, iterations=0, true_relative_inf=0., audits=1)
    dot = lambda a, b: float(xp.sum(a*b, dtype=xp.float64))
    updates = _cg_updates() if xp is not np else None
    z = operator.precondition(r)
    p, rz, audits = z.copy(), dot(r, z), 0
    for iteration in range(1, maxiter+1):
        ap = operator.action(p)
        pap = dot(p, ap)
        if pap <= 0 or rz <= 0 or not np.isfinite(pap+rz):
            raise RuntimeError('HEX8 PCG lost positive curvature')
        alpha = rz/pap
        if updates is None:
            x += alpha*p
            r -= alpha*ap
        else:
            updates[0](p, ap, np.float64(alpha), x, r)
        if float(xp.abs(r).max()) <= threshold:
            x -= x.mean(axis=(1, 2, 3), keepdims=True)
            r = rhs-operator.action(x)
            audits += 1
            relative = float(xp.abs(r).max())/(threshold/rtol)
            if relative <= rtol:
                return dict(solution=x, iterations=iteration, true_relative_inf=relative, audits=audits)
            z = operator.precondition(r)
            p, rz = z.copy(), dot(r, z)
            continue
        z = operator.precondition(r)
        new = dot(r, z)
        if updates is None:
            p *= new/rz
            p += z
        else:
            updates[1](z, np.float64(new/rz), p)
        rz = new
    raise RuntimeError(f'HEX8 PCG failed true residual after {maxiter} iterations')
