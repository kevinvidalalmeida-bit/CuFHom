"""Independent CPU64 screen of full-resolution projected compliance.

No averaging, auxiliary mesh or GenEO. Spectrum/iteration screen only;
the two-operator cost estimate is not a GPU speed measurement.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import cg,LinearOperator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output file')
    shape=(5,5,5);V=int(np.prod(shape));K,G=3.,2.
    wave=np.asarray(np.meshgrid(*[np.fft.fftfreq(n)*n/L for n,L in zip(shape,(2.,3.,4.))],indexing='ij'))
    wave/=np.where(np.linalg.norm(wave,axis=0)>0,np.linalg.norm(wave,axis=0),1.)
    def root(h,a,b):return a*h+(b-a)*wave*np.sum(wave*h,axis=0)
    def strain(u,a,b):
        h=root(np.fft.fftn(u.reshape((3,)+shape),axes=(1,2,3)),a,b)
        z=np.asarray([wave[0]*h[0],wave[1]*h[1],wave[2]*h[2],
                      (wave[1]*h[2]+wave[2]*h[1])/np.sqrt(2),
                      (wave[0]*h[2]+wave[2]*h[0])/np.sqrt(2),
                      (wave[0]*h[1]+wave[1]*h[0])/np.sqrt(2)])
        return np.fft.ifftn(1j*z,axes=(1,2,3)).real
    def adjoint(s,a,b):
        h=np.fft.fftn(s,axes=(1,2,3))
        z=np.asarray([wave[0]*h[0]+(wave[1]*h[5]+wave[2]*h[4])/np.sqrt(2),
                      wave[1]*h[1]+(wave[0]*h[5]+wave[2]*h[3])/np.sqrt(2),
                      wave[2]*h[2]+(wave[0]*h[4]+wave[1]*h[3])/np.sqrt(2)])
        return np.fft.ifftn(-1j*root(z,a,b),axes=(1,2,3)).real.ravel()
    scalar=linalg.null_space(np.ones((1,V)))
    U=linalg.block_diag(scalar,scalar,scalar)
    base=np.zeros((6,6));base[:3,:3]=K-2*G/3
    np.fill_diagonal(base[:3,:3],K+4*G/3);base[3:,3:]=2*G*np.eye(3)
    x,y,z=np.meshgrid(*[(np.arange(n)+.5)/n for n in shape],indexing='ij')
    compact=(x-.5)**2+(y-.5)**2+(z-.5)**2<.3**2
    reference=(1/np.sqrt(G),1/np.sqrt(K+4*G/3))
    compliance=(2*np.sqrt(G),np.sqrt(K+4*G/3))
    probe=np.random.default_rng(15).normal(size=3*V);probe=(U@U.T)@probe
    np.testing.assert_allclose(adjoint(strain(probe,np.sqrt(2),1.),np.sqrt(2),1.),probe,atol=2e-14)
    cases={}
    for contrast in (3.,30.,80.):
        cases[f'compact_{int(contrast)}']=base*np.where(compact,contrast,1.)[...,None,None]
    cases['laminate_30']=base*np.where(x<.4,30.,1.)[...,None,None]
    # General SPD Mandel coefficients, two oblique regions; no fiber geometry.
    rng=np.random.default_rng(218)
    v=rng.normal(size=(2,6));v/=np.linalg.norm(v,axis=1)[:,None]
    Croot=linalg.sqrtm(base)
    tensors=[Croot@(np.eye(6)+79*np.outer(direction,direction))@Croot for direction in v]
    cases['anisotropic_regions_80']=np.where(((x+.6*y+.3*z)%.5<.2)[...,None,None],tensors[0],tensors[1])
    rows=[]
    for name,C in cases.items():
        inverse=np.linalg.inv(C)
        def action(v):return adjoint(np.einsum('...ij,j...->i...',C,strain(v,*reference)),*reference)
        def apply(v):return adjoint(np.einsum('...ij,j...->i...',inverse,strain(v,*compliance)),*compliance)
        H=np.column_stack([U.T@action(column) for column in U.T])
        B=np.column_stack([U.T@apply(column) for column in U.T])
        assert np.linalg.norm(H-H.T)/np.linalg.norm(H)<2e-12
        assert np.linalg.norm(B-B.T)/np.linalg.norm(B)<2e-12
        H=(H+H.T)/2;B=(B+B.T)/2
        values,vectors=linalg.eigh(B);assert values[0]>0
        rootB=(vectors*np.sqrt(values))@vectors.T
        spectra=[linalg.eigvalsh(H),linalg.eigvalsh(rootB@H@rootB)]
        assert spectra[1][0]>=1-2e-11
        counts=[[],[]];errors=[];fields=[]
        for load in range(6):
            E=np.zeros((6,)+shape);E[load]=1
            rhs=-U.T@adjoint(np.einsum('...ij,j...->i...',C,E),*reference)
            for option in range(2):
                count=[0]
                def callback(_):count[0]+=1
                M=LinearOperator(B.shape,matvec=lambda v:B@v) if option else None
                solution,code=cg(H,rhs,M=M,rtol=1e-8,atol=0,maxiter=2000,callback=callback)
                assert code==0
                counts[option].append(count[0])
                errors.append(float(np.linalg.norm(rhs-H@solution)/np.linalg.norm(rhs)))
                if option==0:fields.append(E+strain(U@solution,*reference))
        fields=np.asarray(fields)
        stresses=np.einsum('xyzik,bkxyz->bixyz',C,fields)
        energy=np.einsum('aixyz,bixyz->ab',fields,stresses)/V
        mean_stress=stresses.mean(axis=(2,3,4)).T
        # Exact split of the homogeneous reference on the admissible space.
        delta=C-base
        split=probe+adjoint(np.einsum('...ij,j...->i...',delta,strain(probe,*reference)),*reference)
        split_error=float(np.linalg.norm(split-action(probe))/np.linalg.norm(action(probe)))
        assert split_error<2e-13
        row=dict(case=name,dimension=H.shape[0],current_condition=float(spectra[0][-1]/spectra[0][0]),
                 compliance_condition=float(spectra[1][-1]/spectra[1][0]),
                 compliance_eigen_min=float(spectra[1][0]),compliance_eigen_max=float(spectra[1][-1]),
                 current_iterations=counts[0],compliance_iterations=counts[1],
                 equal_operator_cost_proxy=2*sum(counts[1])/sum(counts[0]),max_coordinate_residual=max(errors),
                 reference_split_relative_error=split_error,
                 mean_stress_Ceff_relative_difference=float(np.linalg.norm(mean_stress-energy)/np.linalg.norm(energy)),
                 mean_stress_symmetry_relative=float(np.linalg.norm(mean_stress-mean_stress.T)/np.linalg.norm(mean_stress)),
                 energy_Ceff_min_eigenvalue=float(linalg.eigvalsh(energy)[0]),
                 mean_stress_Ceff_min_eigenvalue=float(linalg.eigvalsh((mean_stress+mean_stress.T)/2)[0]))
        rows.append(row)
        print(name,round(row['current_condition'],3),'->',round(row['compliance_condition'],3),
              sum(counts[0]),'->',sum(counts[1]),'CG; cost proxy',round(row['equal_operator_cost_proxy'],3),flush=True)
    report=dict(scope='CPU64 complete dense spectrum on odd 5^3, 372 mean-free real coordinates. '
                'Six coordinate RHS, rtol=1e-8. No GPU speed claim or G09 spectrum estimate.',
                preconditioner='R^-1 Q0* C^-1 Q0 R^-1',rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
