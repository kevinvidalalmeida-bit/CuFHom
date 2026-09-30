"""Validated GPU defaults; explicit settings and CPU profiles take precedence."""


def linear_defaults(parameters, *, use_gpu, profile):
    p = dict(parameters)
    if not use_gpu:
        return p
    p.setdefault('cupy_preserve_plans', True)
    p.setdefault('keep_solution_on_device', True)
    p.setdefault('postprocess_batch_size', 1)
    p.setdefault('postprocess_assembly', 'fused')
    dtype = str(p.get('solver_real_dtype', profile.get('solver_real_dtype', 'float32')))
    if dtype != 'float32':
        return p
    p.setdefault('solver_fft_form', 'r')
    p.setdefault('cfield_storage', 'sym21')
    p.setdefault('projection_storage', 'direct')
    p.setdefault('projection_backend', 'numpy')
    p.setdefault('cupy_reuse_inverse_input', True)
    p.setdefault('cupy_fused_xr_rr', True)
    p.setdefault('cupy_fused_dot', True)
    indexed_compatible = (
        p['solver_fft_form'] == 'r'
        and p['cfield_storage'] in ('sym21', 'symmetric21', 'packed21')
        and p['projection_storage'] in ('direct', 'formula', 'elastic_direct')
        and p.get('use_cfield_material_fast_path', True)
        and p.get('cfield_origin', 'zero') in ('zero', '0', 'native', 'grid')
        and not p.get('force_disk_cfield', False)
        and not any(p.get(k) is not None for k in (
            'Cfield_array', 'Cfield_input_path', 'fiber_fraction_array', 'fiber_fraction_path'))
    )
    p.setdefault('cfield_indexed', indexed_compatible)
    return p
