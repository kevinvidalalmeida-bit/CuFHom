"""Reuse must preserve the complete fields and original physical audit."""
import json

import numpy as np
import pytest

from tests.fft.linear.test_linear_coordinates import case,cp,solve_homogenization

# Reexport the autouse fixture so pytest discovers it in this module.
from tests.fft.linear.test_linear_coordinates import gpu as gpu


def run(parameters,path):
    options=dict(parameters,solution_field_return_in_memory=True,solver_timing_path=str(path),
                 linear_fft_autotune=False)
    C=solve_homogenization(options)
    infos=json.loads(path.read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert len(infos)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=parameters['solver_rtol'] for i in infos)
    return C,options['_solution_fields_result'],infos


@pytest.mark.parametrize('shape',[(5,7,5),(6,4,8)])
@pytest.mark.parametrize('fusion',[False,True])
def test_reuse_removes_six_maps_and_preserves_fields(shape,fusion,tmp_path,monkeypatch):
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    original=SymmetricCoordinates.strain
    calls=[]
    def observed(self,value):
        calls.append(value.shape)
        return original(self,value)
    monkeypatch.setattr(SymmetricCoordinates,'strain',observed)
    parameters=dict(case(shape),linear_kernel_fusion=fusion)
    C,fields,base=run(dict(parameters,linear_final_audit_reuse=False),tmp_path/'base.json')
    old=len(calls);calls.clear()
    new,new_fields,infos=run(dict(parameters,linear_final_audit_reuse=True),tmp_path/'new.json')
    assert old-len(calls)==6
    assert all(i['linear_final_strain_reuse'] and i['linear_final_audit_reuse'] for i in infos)
    assert not any(i['linear_final_audit_reuse'] for i in base)
    np.testing.assert_array_equal(new,C)
    for a,b in zip(fields,new_fields):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal([i['true_norm_res_rel'] for i in infos],[i['true_norm_res_rel'] for i in base])


def test_changed_material_warm_start_disables_retention(tmp_path):
    parameters=case((5,7,5))
    _,fields,_=run(parameters,tmp_path/'first.json')
    target=dict(parameters,Em=4.1,Ef_L=82.,initial_solution_fields=fields)
    C,new_fields,infos=run(target,tmp_path/'warm.json')
    baseline,baseline_fields,_=run(dict(target,linear_final_audit_reuse=False),tmp_path/'base.json')
    assert all(i['warm_start_accepted'] for i in infos)
    assert not any(i['linear_final_strain_reuse'] or i['linear_final_audit_reuse'] for i in infos)
    np.testing.assert_array_equal(C,baseline)
    for a,b in zip(new_fields,baseline_fields):np.testing.assert_array_equal(a,b)


def test_false_inner_exit_does_not_reuse_another_iterate(tmp_path,monkeypatch):
    import cufhom.fft.linear.coordinates as lc
    parameters=case((5,7,5))
    expected,_,_=run(dict(parameters,linear_final_audit_reuse=False),tmp_path/'base.json')
    def false_exit(action,rhs,*args,**kwargs):
        kwargs['residual_action'](rhs.copy())
        return cp.zeros_like(rhs),1
    monkeypatch.setattr(lc,'fused_cg',false_exit)
    C,_,infos=run(parameters,tmp_path/'false.json')
    assert not any(i['linear_final_audit_reuse'] for i in infos)
    assert all(i['correction_iterations']>0 for i in infos)
    np.testing.assert_allclose(C,expected,atol=3e-6,rtol=2e-6)


def test_mismatched_physical_operator_keeps_full_audit(monkeypatch,tmp_path):
    from cufhom_core.tensors.operators import Operator
    original=Operator.apply_material_result
    calls=[]
    def mismatch(self,stress,material,projection,**kwargs):
        calls.append(1)
        return original(self,stress,object(),projection,**kwargs)
    monkeypatch.setattr(Operator,'apply_material_result',mismatch)
    _,_,infos=run(case((5,7,5)),tmp_path/'mismatch.json')
    assert len(calls)==6
    assert all(i['linear_final_strain_reuse'] and not i['linear_final_audit_reuse'] for i in infos)


@pytest.mark.parametrize('shape,policy',[((5,7,5),'full'),((6,4,8),'component'),
                                       ((6,4,8),'batch2'),((6,4,8),'batch3')])
def test_spectrum_reuse_removes_six_RFFT_calls(shape,policy,tmp_path,monkeypatch):
    from cufhom_core.tensors import operators
    original=operators.rfftn_cached
    calls=[]
    def observed(*args):
        calls.append(1);return original(*args)
    monkeypatch.setattr(operators,'rfftn_cached',observed)
    parameters=dict(case(shape),cupy_rfft_batch_mode=policy)
    C,fields,base=run(dict(parameters,linear_final_spectrum_reuse=False),tmp_path/'base.json')
    old=len(calls);calls.clear()
    new,new_fields,infos=run(parameters,tmp_path/'new.json')
    assert old-len(calls)==6
    assert all(i['linear_final_spectrum_reuse'] for i in infos)
    assert not any(i['linear_final_spectrum_reuse'] for i in base)
    np.testing.assert_array_equal(new,C)
    for a,b in zip(fields,new_fields):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal([i['true_norm_res_rel'] for i in infos],[i['true_norm_res_rel'] for i in base])


@pytest.mark.parametrize('invalid',['shape','dtype','scaling'])
def test_incompatible_spectrum_retains_full_audit(invalid,tmp_path,monkeypatch):
    from cufhom_core.tensors import operators
    original=operators.Operator.apply_material_result
    reference,fields,_=run(case((5,7,5)),tmp_path/'base.json')
    calls=[]
    def incompatible(self,stress,material,projection,**kwargs):
        spectrum=kwargs['spectrum'];calls.append(1)
        if invalid=='shape':kwargs['spectrum']=spectrum[:1]
        elif invalid=='dtype':kwargs['spectrum']=spectrum.astype(cp.complex128)
        else:
            with monkeypatch.context() as m:
                m.setattr(operators,'_CUPY_UNSCALED_FFT_PAIR',False)
                return original(self,stress,material,projection,**kwargs)
        return original(self,stress,material,projection,**kwargs)
    monkeypatch.setattr(operators.Operator,'apply_material_result',incompatible)
    C,new_fields,infos=run(case((5,7,5)),tmp_path/'new.json')
    assert len(calls)==6 and not any(i['linear_final_spectrum_reuse'] or i['linear_final_audit_reuse'] for i in infos)
    np.testing.assert_array_equal(C,reference)
    for a,b in zip(fields,new_fields):np.testing.assert_array_equal(a,b)


def test_profile_skips_only_material_and_forward(tmp_path,monkeypatch):
    from cufhom_core.tensors.operators import Operator
    original=Operator.apply_material_result;profiles=[]
    def profiled(self,*args,**kwargs):
        stats={name:dict(seconds=0.,calls=0) for name in ['A','FN','hG','FiN']}
        self._component_profile=stats
        try:return original(self,*args,**kwargs)
        finally:
            profiles.append(stats);del self._component_profile
    monkeypatch.setattr(Operator,'apply_material_result',profiled)
    _,_,infos=run(case((5,7,5)),tmp_path/'new.json')
    assert len(profiles)==6 and all(i['linear_final_spectrum_reuse'] for i in infos)
    for p in profiles:
        assert p['A']['calls']==p['FN']['calls']==0
        assert p['hG']['calls']==p['FiN']['calls']==1
