"""Double-float (df64) J2 returns against the IEEE FP64 kernel, GPU only."""

import numpy as np
import pytest
from cufhom.materials.plasticity import CuPyMaterial, J2Material, PlasticState

cp = pytest.importorskip("cupy")

LAWS = dict(
    voce=dict(
        E=3.76,
        nu=0.39,
        yield_stress=0.029,
        hardening=0.05,
        saturation_stress=0.06,
        saturation_exponent=100.0,
    ),
    kinematic=dict(
        E=3.76,
        nu=0.39,
        yield_stress=0.029,
        hardening=0.02,
        saturation_stress=0.045,
        saturation_exponent=100.0,
        kinematic_hardening=0.3,
    ),
    linear=dict(E=3.76, nu=0.39, yield_stress=0.029, hardening=0.05),
)


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("law", list(LAWS))
@pytest.mark.parametrize("midpoint", [True, False])
def test_df64_return_matches_ieee_fp64(law, midpoint):
    shape = (16, 16, 16)
    ids = cp.zeros(shape, cp.int32)
    ids[:, :, :3] = 1
    rng = cp.random.default_rng(3)
    e = rng.standard_normal((6,) + shape) * 0.01
    state = PlasticState(
        rng.standard_normal((6,) + shape) * 1e-3,
        cp.abs(rng.standard_normal(shape)) * 1e-2,
    )
    table = cp.asarray(np.tile(np.eye(6) * 30, (2, 1, 1)))
    m = J2Material(**LAWS[law])
    responses = [
        CuPyMaterial(cp, m, ids, table, "float64", arithmetic).update(
            e, state, 0.9 * e if midpoint else None
        )
        for arithmetic in ("ieee", "df64")
    ]
    assert float(cp.max(responses[0].dp)) > 0
    for name in ("stress", "direction", "a", "b", "dp"):
        ieee, df = getattr(responses[0], name), getattr(responses[1], name)
        assert float(cp.max(cp.abs(df - ieee))) <= 1e-12 * float(cp.max(cp.abs(ieee)))
    v = rng.standard_normal((6,) + shape)
    t = [
        CuPyMaterial(cp, m, ids, table, "float64", a).tangent(v, responses[0])
        for a in ("ieee", "df64")
    ]
    assert float(cp.max(cp.abs(t[1] - t[0]))) <= 1e-12 * float(cp.max(cp.abs(t[0])))


def test_df64_exp_and_arithmetic_contract():
    from cufhom.materials.double_float import df64_source

    code = df64_source(r"""
extern "C" __global__ void k(const MEM* x,MEM* y,long long n){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const REAL v=L(x[i]);y[i]=S(exp(v)*(REAL)1.5);}""")
    kernel = cp.RawKernel(code, "k")
    # The df input carries ~47-48 bits: exp's error is below |x| 2^-47 (8.5e-14 here).
    x = cp.linspace(-12.0, 12.0, 4097)
    y = cp.empty_like(x)
    kernel((17,), (256,), (x, y, np.int64(x.size)))
    np.testing.assert_allclose(cp.asnumpy(y), 1.5 * np.exp(cp.asnumpy(x)), rtol=1e-13)
    with pytest.raises(ValueError):
        CuPyMaterial(
            cp,
            J2Material(),
            cp.zeros((2, 2, 2), cp.int32),
            cp.eye(6)[None],
            "float32",
            "df64",
        )


def test_df64_paraboloidal_return_matches_ieee_fp64():
    from cufhom.materials.paraboloidal import CuPyParaboloidal, ParaboloidalMaterial

    shape = (16, 16, 16)
    ids = cp.zeros(shape, cp.int32)
    ids[:, :, :3] = 1
    rng = cp.random.default_rng(5)
    e = rng.standard_normal((6,) + shape) * 0.02
    state = PlasticState(
        rng.standard_normal((6,) + shape) * 1e-3,
        cp.abs(rng.standard_normal(shape)) * 1e-2,
    )
    table = cp.asarray(np.tile(np.eye(6) * 30, (2, 1, 1)))
    m = ParaboloidalMaterial(
        tension_curve=[[0, 0.029], [0.02, 0.05], [0.1, 0.07]],
        compression_curve=[[0, 0.067], [0.2, 0.12]],
    )
    responses = [
        CuPyParaboloidal(cp, m, ids, table, "float64", a).update(e, state)
        for a in ("ieee", "df64")
    ]
    assert float(cp.max(responses[0].dp)) > 0
    for name in (
        "stress",
        "direction",
        "a",
        "b",
        "dp",
        "bulk",
        "coupling_to_direction",
        "coupling_to_pressure",
    ):
        ieee, df = getattr(responses[0], name), getattr(responses[1], name)
        assert float(cp.max(cp.abs(df - ieee))) <= 1e-11 * float(cp.max(cp.abs(ieee)))
