"""Independent analytic laminate and calibrated-point checks on GPU."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization

pytestmark = pytest.mark.gpu


def test_elastic_periodic_laminate_shear_has_harmonic_effective_modulus():
    shape = (11, 11, 11)  # Odd grid: no excluded Nyquist plane in the analytic oracle.
    phase = np.zeros(shape, np.uint8)
    phase[3:6] = 1
    ori = np.zeros(shape + (3,))
    ori[..., 0] = 1
    mu_matrix = 3 / (2 * (1 + 0.3))
    mu_fiber = 5.0
    fraction = phase.mean()
    expected = 1 / ((1 - fraction) / mu_matrix + fraction / mu_fiber)
    loading = dict(
        strain_control=[True] * 6,
        strain=[[0.0] * 6, [0, 0, 0, 0, 0, 0.001]],
        stress=[[0.0] * 6, [0.0] * 6],
    )
    r = solve_nonlinear_homogenization(
        dict(
            phase_array=phase,
            ori_array=ori,
            backend="cupy",
            model="drucker_prager",
            loading=loading,
            initial_increment=1.0,
            dtype="float64",
            rtol=1e-9,
        )
    )
    # sigma_M = 2*G_eff*epsilon_M for shear. Layers normal to x,
    # tangential displacement in y => constant shear traction and series law.
    actual = r["history"][-1]["stress"][5] / (0.002)
    assert actual == pytest.approx(expected, rel=2e-8)


def test_literature_card_reproduces_idealized_uniaxial_yield_points():
    root = Path(__file__).resolve().parents[3]
    mat = json.loads(
        (root / "examples/material_resin_l_dp_approximation.json").read_text()
    )
    m = mat["matrix"]
    a = m["friction"]
    sy = m["yield_stress"]
    ft, fc = sy / (1 + a / 3), sy / (1 - a / 3)
    assert ft == pytest.approx(0.039)
    assert fc == pytest.approx(0.081)
    shape = (4, 4, 4)
    for signed_yield in (ft, -fc):
        endpoint = signed_yield / m["E"]
        r = solve_nonlinear_homogenization(
            dict(
                phase_array=np.zeros(shape, np.uint8),
                ori_array=np.zeros(shape + (3,)),
                matrix=m,
                fiber=mat["fiber"],
                backend="cupy",
                model="drucker_prager",
                dtype="float64",
                rtol=1e-9,
                path=[0.0, endpoint],
                initial_increment=abs(endpoint),
            )
        )
        assert r["history"][-1]["stress"][0] == pytest.approx(signed_yield, rel=2e-8)
        assert r["history"][-1]["hardening_variable_mean"] < 1e-9
