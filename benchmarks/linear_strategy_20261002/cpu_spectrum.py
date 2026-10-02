"""Dense CPU64 fine spectrum: independent unitary real displacement basis."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'descartados/auxiliary_geneo_20261002'))
import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import cg,LinearOperator
from auxiliary_space import element_matrices,assemble,Multigrid,geneo_basis,project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new output file')
    shape=(5,5,5);V=np.prod(shape);lengths=(2.,3.,4.);K0=3.;G0=2.
    f=[np.fft.fftfreq(n)*n/L for n,L in zip(shape,lengths)]
    k=np.asarray(np.meshgrid(*f,indexing='ij'));kn=np.linalg.norm(k,axis=0)
    n=k/np.where(kn>0,kn,1);n[:,0,0,0]=0
    transform=lambda x:np.fft.fftn(x.reshape((3,)+shape),axes=(1,2,3))
    inverse=lambda h:np.fft.ifftn(h,axes=(1,2,3)).real
    def root(h,a,b):return a*h+(b-a)*n*np.sum(n*h,axis=0)
    def strain(y):
        h=root(transform(y),1/np.sqrt(G0),1/np.sqrt(K0+4*G0/3))
        z=np.asarray([n[0]*h[0],n[1]*h[1],n[2]*h[2],
            (n[1]*h[2]+n[2]*h[1])/np.sqrt(2),
            (n[0]*h[2]+n[2]*h[0])/np.sqrt(2),
            (n[0]*h[1]+n[1]*h[0])/np.sqrt(2)])
        return np.fft.ifftn(1j*z,axes=(1,2,3)).real
    def adjoint(stress):
        h=np.fft.fftn(stress,axes=(1,2,3))
        z=np.asarray([n[0]*h[0]+(n[1]*h[5]+n[2]*h[4])/np.sqrt(2),
            n[1]*h[1]+(n[0]*h[5]+n[2]*h[3])/np.sqrt(2),
            n[2]*h[2]+(n[0]*h[4]+n[1]*h[3])/np.sqrt(2)])
        return inverse(-1j*root(z,1/np.sqrt(G0),1/np.sqrt(K0+4*G0/3))).ravel()
    def transfer(u):
        h=root(transform(u),np.sqrt(G0),np.sqrt(K0+4*G0/3))
        return inverse(2*np.pi*kn*h).ravel()
    scalar=linalg.null_space(np.ones((1,V)));U=sparse.block_diag([scalar]*3).toarray()
    base=np.zeros((6,6));base[:3,:3]=K0-2*G0/3
    np.fill_diagonal(base[:3,:3],K0+4*G0/3);base[3:,3:]=2*G0*np.eye(3)
    x,y,z=np.meshgrid(*[(np.arange(n)+.5)/n for n in shape],indexing='ij')
    rows=[]
    for family in ['compact','laminate']:
        phase=((x-.5)**2+(y-.5)**2+(z-.5)**2<.3**2) if family=='compact' else (x<.4)
        C=base[None,None,None]*np.where(phase,30.,1.)[...,None,None]
        def H(v):return adjoint(np.einsum('...ij,j...->i...',C,strain(v)))
        fine=np.column_stack([U.T@H(v) for v in U.T])
        assert np.linalg.norm(fine-fine.T)/np.linalg.norm(fine)<2e-12
        fine=(fine+fine.T)*.5
        elements=element_matrices(C.reshape(-1,6,6),lengths,shape)*V/np.prod(lengths)
        Ka=assemble(elements,shape);S=Multigrid(Ka,shape)
        Z,inv,detail=geneo_basis(Ka,elements,shape)
        def enriched(v):return S(v)+project(Z@(inv@(Z.T@project(v))))
        maps={'current':lambda v:v,'auxiliary':lambda v:v+transfer(S(transfer(v))),
              'geneo':lambda v:v+transfer(enriched(transfer(v)))}
        lifted=np.column_stack([transfer(Z[:,j].toarray().ravel()) for j in range(Z.shape[1])])
        coarse=(U.T@lifted).T@fine@(U.T@lifted)
        values,vectors=linalg.eigh((coarse+coarse.T)*.5);valid=values>values[-1]*1e-10
        coarse_inverse=(vectors[:,valid]/values[valid])@vectors[:,valid].T
        maps['fine_geneo']=lambda v:v+lifted@(coarse_inverse@(lifted.T@v))
        spectra={}
        for name,apply in maps.items():
            B=np.column_stack([U.T@apply(v) for v in U.T])
            symmetry=float(np.linalg.norm(B-B.T)/np.linalg.norm(B));assert symmetry<2e-12
            values,vectors=linalg.eigh((B+B.T)*.5);assert values[0]>0
            rootB=(vectors*np.sqrt(values))@vectors.T
            eigenvalues=linalg.eigvalsh(rootB@fine@rootB)
            counts=[];errors=[]
            for load in range(6):
                E=np.zeros((6,)+shape);E[load]=1
                rhs=-U.T@adjoint(np.einsum('...ij,j...->i...',C,E))
                iterations=[0]
                def callback(_):iterations[0]+=1
                solution,code=cg(fine,rhs,M=LinearOperator(B.shape,matvec=lambda v:B@v),
                                  rtol=1e-8,atol=0.,maxiter=2000,callback=callback)
                assert code==0
                counts.append(iterations[0]);errors.append(float(np.linalg.norm(rhs-fine@solution)/max(np.linalg.norm(rhs),1e-30)))
            spectra[name]=dict(condition=float(eigenvalues[-1]/eigenvalues[0]),
                eigen_min=float(eigenvalues[0]),eigen_max=float(eigenvalues[-1]),
                B_min=float(values[0]),symmetry=symmetry,iterations=counts,max_coordinate_residual=max(errors))
        rows.append(dict(family=family,shape=shape,real_fine_dimension=fine.shape[0],spectra=spectra,geneo=detail))
    report=dict(scope='Independent CPU64 dense fine spectrum, all mean-free real modes of odd 5^3. '
        'Local Q1 auxiliary, not a hardware timing. Six CG RHS use coordinate rtol 1e-8 for this spectral check only.',rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({row['family']:{name:(v['condition'],sum(v['iterations'])) for name,v in row['spectra'].items()} for row in rows})


if __name__=='__main__':main()
