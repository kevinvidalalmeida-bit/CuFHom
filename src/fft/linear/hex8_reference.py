"""Staged, fully integrated HEX8 action for controlled fusion ablations.

``Hex8ReferenceOperator`` inherits the indexed materials, geometry, reference
inverse, RHS, mean stress and solver contract from ``hex8_general.Hex8Operator``.
Only the stiffness action changes: each of the eight Gauss points materializes
``strain = B_g u`` and ``stress = C_e strain`` before the adjoint gathers forces.
One Gauss point is live at a time (two six-component global fields), with no
element stiffness contraction, global dense matrix, or atomic force scatter.

Use ``hex8_general.solve_hex8_load(op, macro, ...)`` directly, or select this
class with ``homogenize_hex8(..., operator_type=Hex8ReferenceOperator)``. The
six-load/storage workflow and PCG are shared with the optimized action.
"""

import numpy as np

from .hex8_general import _NODES, Hex8Operator, _kernel

_CUDA_STRAIN = r"""
extern "C" __global__ void hex8_reference_strain(
 const REAL* u,const REAL* b,REAL* strain,long long n,int nx,int ny,int nz){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
 REAL e[6]={0,0,0,0,0,0};
 // Partition of unity: B_0 u_0 = -sum_{a>0} B_a u_0.
 for(int a=1;a<8;a++){
  int xx=(x+((a>>2)&1))%nx,yy=(y+((a>>1)&1))%ny,zz=(z+(a&1))%nz;
  long long j=((long long)xx*ny+yy)*nz+zz;
  for(int s=0;s<6;s++)for(int c=0;c<3;c++)
   e[s]+=b[(s*8+a)*3+c]*(u[c*n+j]-u[c*n+i]);
 }
 for(int s=0;s<6;s++)strain[s*n+i]=e[s];
}
"""

_CUDA_CONSTITUTIVE = r"""
extern "C" __global__ void hex8_reference_constitutive(
 const int* index,const REAL* c,const REAL* strain,REAL* stress,long long n){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 int p=index[i];
 for(int s=0;s<6;s++){
  REAL value=0;
  for(int t=0;t<6;t++)value+=c[(p*6+s)*6+t]*strain[t*n+i];
  stress[s*n+i]=value;
 }
}
"""

_CUDA_ADJOINT = r"""
extern "C" __global__ void hex8_reference_adjoint(
 const REAL* stress,const REAL* b,REAL weight,REAL* out,
 long long n,int nx,int ny,int nz){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
 REAL f[3]={0,0,0};
 // Transpose of the anchored gradient: periodic gather minus local force.
 for(int a=1;a<8;a++){
  int xx=(x-((a>>2)&1)+nx)%nx,yy=(y-((a>>1)&1)+ny)%ny,zz=(z-(a&1)+nz)%nz;
  long long j=((long long)xx*ny+yy)*nz+zz;
  for(int c=0;c<3;c++)for(int s=0;s<6;s++)
   f[c]+=b[(s*8+a)*3+c]*(stress[s*n+j]-stress[s*n+i]);
 }
 for(int c=0;c<3;c++)out[c*n+i]+=weight*f[c];
}
"""


class Hex8ReferenceOperator(Hex8Operator):
    """Unfused ``sum_g B_g.T C_e B_g / (8*n)`` on NumPy or CuPy.

    All constructor arguments and inherited storage use the parent convention.
    Integration is normalized by domain volume, so each weight is ``1/(8*n)``
    even for nonunit lengths. Partition of unity is evaluated through nodal
    differences and its exact transpose, retaining exact translation null modes.
    ``audit=True`` uses FP64 gradients and the parent's precision-rounded
    material table, independently of its precontracted element matrices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        xp = self.xp
        self._reference_b = xp.asarray(self.B, dtype=self.working, order="C")
        self._reference_c = xp.asarray(self.C, dtype=self.working, order="C")
        self._reference_b64 = (
            self._reference_b
            if self.working == np.float64
            else xp.asarray(self.B, dtype=xp.float64, order="C")
        )
        self._reference_c64 = (
            self._reference_c
            if self.working == np.float64
            else xp.asarray(self.C, dtype=xp.float64, order="C")
        )
        self._workspaces = {}
        if xp is not np:
            self._reference_kernels = {}
            for dtype in (self.working, np.dtype("float64")):
                if dtype not in self._reference_kernels:
                    self._reference_kernels[dtype] = tuple(
                        _kernel(source, name, dtype)
                        for source, name in (
                            (_CUDA_STRAIN, "hex8_reference_strain"),
                            (_CUDA_CONSTITUTIVE, "hex8_reference_constitutive"),
                            (_CUDA_ADJOINT, "hex8_reference_adjoint"),
                        )
                    )

    def action(self, u, *, audit=False, out=None):
        """Evaluate the staged action; output validation matches the parent."""
        xp, dtype = self.xp, np.dtype("float64") if audit else self.working
        u = xp.ascontiguousarray(u, dtype=dtype)
        if u.shape != (3, *self.shape):
            raise ValueError("Expected three displacement components on the X,Y,Z grid")
        out = xp.empty_like(u) if out is None else out
        if (
            out.shape != u.shape
            or out.dtype != u.dtype
            or not out.flags.c_contiguous
            or xp.may_share_memory(u, out)
        ):
            raise ValueError(
                "action output must be contiguous, correctly typed and distinct from its input"
            )
        b = self._reference_b64 if audit else self._reference_b
        c = self._reference_c64 if audit else self._reference_c
        key = np.dtype(dtype)
        if key not in self._workspaces:
            strain = xp.empty((6, *self.shape), dtype=dtype)
            self._workspaces[key] = (strain, xp.empty_like(strain))
        strain, stress = self._workspaces[key]
        weight = dtype.type(1 / (8 * self.n))
        out.fill(0)
        if xp is np:
            for bg in b:
                self._numpy_strain(u, bg, strain)
                self._numpy_constitutive(c, strain, stress)
                self._numpy_adjoint(stress, bg, weight, out)
        else:
            gradient, constitutive, adjoint = self._reference_kernels[dtype]
            grid, block = ((self.n + 127) // 128,), (128,)
            dims = (np.int64(self.n), *map(np.int32, self.shape))
            for g in range(8):
                bg = b[g]
                gradient(grid, block, (u, bg, strain, *dims))
                constitutive(
                    grid, block, (self.index, c, strain, stress, np.int64(self.n))
                )
                adjoint(grid, block, (stress, bg, weight, out, *dims))
        return out

    @staticmethod
    def _numpy_strain(u, b, strain):
        strain.fill(0)
        for a, node in enumerate(_NODES[1:], 1):
            values = np.roll(u, tuple(-node), (1, 2, 3)) - u
            # Only nine nonzero entries of the Mandel gradient per corner.
            for s, component in zip(*np.nonzero(b[:, a])):
                strain[s] += b[s, a, component] * values[component]

    def _numpy_constitutive(self, c, strain, stress):
        flat, result, labels = (
            strain.reshape(6, -1),
            stress.reshape(6, -1),
            self.index.ravel(),
        )
        # A bounded gather, never a global (n,6,6) constitutive matrix field.
        for start in range(0, self.n, 8192):
            sl = slice(start, min(start + 8192, self.n))
            np.einsum("nij,jn->in", c[labels[sl]], flat[:, sl], out=result[:, sl])

    @staticmethod
    def _numpy_adjoint(stress, b, weight, out):
        force = np.empty_like(out)
        for a, node in enumerate(_NODES[1:], 1):
            np.einsum("sc,s...->c...", b[:, a], stress, out=force)
            out += weight * (np.roll(force, tuple(node), (1, 2, 3)) - force)
