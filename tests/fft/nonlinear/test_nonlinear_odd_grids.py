"""Nonlinear Newton solves on odd grids: the fused tangent pipeline (line-pair KC)
against the classic coordinate route, for each compact-tangent variant."""

import numpy as np
import pytest
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization

cp = pytest.importorskip("cupy")

LAWS = {
    "j2": dict(
        model="j2",
        matrix=dict(
            E=3.0,
            nu=0.3,
            yield_stress=0.02,
            hardening=0.1,
            saturation_stress=0.04,
            saturation_exponent=50.0,
        ),
    ),
    "drucker_prager": dict(
        model="drucker_prager",
        matrix=dict(
            E=3.0, nu=0.3, yield_stress=0.02, hardening=0.1, friction=0.3, dilatancy=0.1
        ),
    ),
    "paraboloidal": dict(
        model="paraboloidal",
        matrix=dict(
            E=3.0,
            nu=0.35,
            plastic_poisson=0.3,
            tension_curve=[[0, 0.02], [0.05, 0.04]],
            compression_curve=[[0, 0.04], [0.05, 0.07]],
        ),
    ),
}


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("law", list(LAWS))
def test_odd_grid_pipeline_matches_classic(law):
    shape = (9, 7, 15)
    phase = np.zeros(shape, np.uint8)
    phase[2:5, 1:4, :] = 1
    phase[6:8, 4:6, 3:9] = 1
    ori = np.zeros(shape + (3,))
    ori[..., 2] = 1
    p = dict(
        phase_array=phase,
        ori_array=ori,
        backend="cupy",
        dtype="float32",
        path=[0.0, 0.012],
        initial_increment=0.002,
        rtol=1e-6,
        return_fields=True,
        **LAWS[law],
    )
    a = solve_nonlinear_homogenization(dict(p, coordinate_pcg="pipeline"))
    b = solve_nonlinear_homogenization(dict(p, coordinate_pcg="classic"))
    assert a["coordinate_pcg"] == "pipeline" and b["coordinate_pcg"] == "classic"
    assert max(r["plastic_fraction"] for r in a["history"]) > 0
    sa, sb = (
        np.array([r["stress"] for r in a["history"]]),
        np.array([r["stress"] for r in b["history"]]),
    )
    np.testing.assert_allclose(sa, sb, rtol=0, atol=2e-5 * np.abs(sb).max())
