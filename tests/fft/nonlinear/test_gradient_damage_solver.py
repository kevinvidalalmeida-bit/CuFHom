"""Coupled matrix-damage acceptance, restart and route checks, GPU only."""

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization
from cufhom.materials.gradient_damage import CuPyGradientDamage

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


def parameters(**changes):
    phase = np.zeros((8, 8, 8), np.uint8)
    ori = np.zeros(phase.shape + (3,))
    ori[..., 0] = 1
    p = dict(
        phase_array=phase,
        ori_array=ori,
        model="drucker_prager",
        backend="cupy",
        dtype="float32",
        matrix=dict(
            E=3.0, nu=0.3, yield_stress=0.3, hardening=0.03, friction=0.3, dilatancy=0.3
        ),
        path=[0.0, 0.024],
        initial_increment=0.002,
        rtol=3e-5,
        return_fields=True,
        damage=dict(
            strain_threshold=0.008,
            softening_strain=0.012,
            length_scale=0.04,
            strength_ratio=2.0,
            residual_stiffness=0.001,
        ),
    )
    p.update(changes)
    return p


def test_uniform_damage_curve_matches_closed_form_and_unloading():
    p = parameters(path=[0.0, 0.024, 0.0])
    result = solve_nonlinear_homogenization(p)
    running = 0.0
    for row in result["history"]:
        e = row["strain"][0]
        running = max(running, e)
        d = (
            1 - 0.008 / running * np.exp(-(running - 0.008) / 0.012)
            if running > 0.008
            else 0.0
        )
        g = 1 - (1 - 0.001) * d
        assert row["damage_mean"] == pytest.approx(d, abs=3e-5)
        assert row["stress"][0] == pytest.approx(g * 3 * e, rel=2e-4, abs=1e-7)
        assert row["damage_gap"] <= p["damage"].get("rtol", 1e-5)
        assert row["damage_dissipation"] >= 0
        assert row["plastic_dissipation"] == 0
    assert result["history"][-1]["damage_mean"] > 0.8
    assert np.linalg.norm(result["history"][-1]["stress"]) < 1e-6
    assert result["damage"]["model"] == "implicit_gradient"


def test_restart_retains_damage_history_and_matches_continuation(tmp_path):
    checkpoint = tmp_path / "confirmed.npz"
    prefix = solve_nonlinear_homogenization(
        parameters(path=[0.0, 0.012], checkpoint_path=checkpoint)
    )
    resumed = solve_nonlinear_homogenization(
        parameters(path=[0.012, 0.024], restart_from=checkpoint)
    )
    complete = solve_nonlinear_homogenization(parameters())
    assert resumed["history"][: len(prefix["history"])] == prefix["history"]
    for key in ("strain", "stress", "damage", "damage_history"):
        np.testing.assert_allclose(
            resumed["fields"][key], complete["fields"][key], rtol=3e-4, atol=3e-6
        )
    different = parameters(path=[0.012, 0.024], restart_from=checkpoint)
    different["damage"]["length_scale"] = 0.05
    with pytest.raises(ValueError, match="mismatch"):
        solve_nonlinear_homogenization(different)


def test_rejected_damage_iteration_cannot_age_confirmed_state(monkeypatch):
    original = CuPyGradientDamage.propose
    snapshots = []

    def fail_once(self, e):
        proposed, info = original(self, e)
        snapshots.append(cp.asnumpy(self.confirmed.value))
        if len(snapshots) == 1:
            self.set_trial(proposed)
            raise NonlinearConvergenceError("injected unconfirmed damage trial")
        return proposed, info

    monkeypatch.setattr(CuPyGradientDamage, "propose", fail_once)
    result = solve_nonlinear_homogenization(
        parameters(path=[0.0, 0.012], initial_increment=0.012, globalization="newton")
    )
    assert result["rejected_steps"]
    np.testing.assert_array_equal(snapshots[0], 0)
    np.testing.assert_array_equal(snapshots[1], 0)
    assert all(np.diff([r["damage_mean"] for r in result["history"]]) >= -1e-7)


def test_damage_pipeline_and_classic_match_for_plastic_composite():
    # Softening makes the coupled increment non-unique for large steps (1e-3
    # and 2e-4 converge to different damage states on each route); 1e-4 does not.
    p = parameters(path=[0.0, 0.008], initial_increment=0.0001)
    p["phase_array"][3:5] = 1
    p["matrix"].update(yield_stress=0.015, dilatancy=0.1)
    p["damage"].update(
        strain_threshold=0.004, softening_strain=0.012, length_scale=0.08
    )
    a = solve_nonlinear_homogenization(dict(p, coordinate_pcg="classic"))
    b = solve_nonlinear_homogenization(dict(p, coordinate_pcg="pipeline"))
    assert a["linear_solver"] == b["linear_solver"] == "bicgstab"
    assert b["history"][-1]["plastic_fraction"] > 0
    assert b["history"][-1]["damage_mean"] > 0
    for key in ("strain", "stress", "damage"):
        np.testing.assert_allclose(
            a["fields"][key], b["fields"][key], rtol=3e-3, atol=1e-5
        )
    np.testing.assert_array_equal(b["fields"]["damage"][p["phase_array"] == 1], 0)


@pytest.mark.parametrize(
    "globalization,working",
    [("newton", "float32"), ("ptc", "float32"), ("newton", "float64")],
)
def test_fp64_coupled_general_loading_checkpoint(tmp_path, globalization, working):
    loading = dict(
        strain_control=[True, True, True, False, False, True],
        parameter=[0.0, 1.0],
        strain=[[0.0] * 6, [0.012, -0.003, -0.003, 0.0, 0.0, 0.002]],
        stress=[[0.0] * 6, [0.0] * 6],
    )
    p = parameters(
        dtype="float64",
        globalization=globalization,
        loading=loading,
        working=working,
        initial_increment=0.5,
        rtol=1e-7,
        checkpoint_path=tmp_path / "mixed.npz",
    )
    p.pop("path")
    p["damage"].update(rtol=1e-8, helmholtz_rtol=1e-10)
    result = solve_nonlinear_homogenization(p)
    # FP64 state and acceptance; the default solves run on the FP32 pipeline.
    assert result["coordinate_pcg"] == (
        "pipeline" if working == "float32" else "classic"
    )
    assert result["working"] == working
    assert result["history"][-1]["damage_mean"] > 0
    for row in result["history"]:
        assert row["residual"] <= 1e-7
        assert row["macro_strain_residual"] <= 1e-7
        assert row["damage_gap"] <= 1e-8
    continuation = dict(
        loading,
        strain=[loading["strain"][-1], [0.014, -0.003, -0.003, 0.0, 0.0, 0.003]],
        parameter=[1.0, 2.0],
    )
    p.update(loading=continuation, restart_from=tmp_path / "mixed.npz")
    resumed = solve_nonlinear_homogenization(p)
    assert resumed["history"][: len(result["history"])] == result["history"]
    assert resumed["history"][-1]["target_parameter"] == 2.0


@pytest.mark.parametrize(
    "changes", [dict(backend="scipy"), dict(model="j2"), dict(elastic_segments=True)]
)
def test_unsupported_damage_contract_is_explicit(changes):
    with pytest.raises(ValueError, match="damage|Damage|Drucker|elastic"):
        solve_nonlinear_homogenization(parameters(**changes))


def test_paraboloidal_damage_uniform_curve_matches_closed_form():
    matrix = dict(
        E=3.0,
        nu=0.3,
        plastic_poisson=0.3,
        tension_curve=[[0, 10.0]],
        compression_curve=[[0, 20.0]],
    )
    result = solve_nonlinear_homogenization(
        parameters(model="paraboloidal", matrix=matrix, path=[0.0, 0.024, 0.0])
    )
    running = 0.0
    for row in result["history"]:
        e = row["strain"][0]
        running = max(running, e)
        d = (
            1 - 0.008 / running * np.exp(-(running - 0.008) / 0.012)
            if running > 0.008
            else 0.0
        )
        g = 1 - (1 - 0.001) * d
        assert row["damage_mean"] == pytest.approx(d, abs=3e-5)
        assert row["stress"][0] == pytest.approx(g * 3 * e, rel=2e-4, abs=1e-7)
        assert row["damage_dissipation"] >= 0
    assert (
        result["damage"]["law"]
        == "effective_stress_paraboloidal_exponential_nonlocal_total_strain"
    )
    assert result["damage"]["interface"] == "perfectly_bonded"


def test_interphase_with_bulk_parameters_reproduces_bonded_damage():
    p = parameters(path=[0.0, 0.008], initial_increment=0.001)
    p["phase_array"][3:5] = 1
    p["matrix"].update(yield_stress=0.015, dilatancy=0.1)
    p["damage"].update(
        strain_threshold=0.004, softening_strain=0.012, length_scale=0.08
    )
    bonded = solve_nonlinear_homogenization(p)
    same = dict(
        p,
        damage=dict(
            p["damage"],
            interphase_thickness=0.125,
            interphase_strain_threshold=0.004,
            interphase_softening_strain=0.012,
        ),
    )
    weak = dict(p, damage=dict(same["damage"], interphase_strain_threshold=0.002))
    a, b = solve_nonlinear_homogenization(same), solve_nonlinear_homogenization(weak)
    np.testing.assert_array_equal(a["fields"]["damage"], bonded["fields"]["damage"])
    assert a["history"][-1]["interphase_fraction"] == pytest.approx(0.25)
    assert (
        b["history"][-1]["damage_interphase_mean"]
        > a["history"][-1]["damage_interphase_mean"]
    )
    assert b["history"][-1]["stress"][0] < a["history"][-1]["stress"][0]
    assert b["damage"]["interface"] == "damaging_interphase"
