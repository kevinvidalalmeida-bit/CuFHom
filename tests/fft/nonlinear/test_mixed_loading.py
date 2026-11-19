"""Mixed Mandel controls: host validation and exclusively CuPy solves.

GPU cases are meant to be run serially by the coordinating process. They are
not run as part of preparing this change.
"""

import copy
import csv
import json

import numpy as np
import pytest
from cufhom.fft.nonlinear import solver
from cufhom.fft.nonlinear.krylov import TangentPipeline
from cufhom.fft.nonlinear.loading import MixedLoadingPath
from cufhom.materials.drucker_prager import CuPyDruckerPrager, DruckerPragerMaterial
from cufhom.materials.plasticity import initial_state

MASKS = [
    [True] * 6,
    [False] * 6,
    [True, False, True, False, True, False],
    [False, True, False, False, False, True],
]


def loading(mask=None, strain=None, stress=None, parameter=None):
    value = dict(
        strain_control=[True, False, False, False, False, False]
        if mask is None
        else mask,
        strain=[[0.0] * 6, [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]]
        if strain is None
        else strain,
        stress=[[0.0] * 6, [0.0] * 6] if stress is None else stress,
    )
    if parameter is not None:
        value["parameter"] = parameter
    return value


@pytest.fixture
def gpu():
    cp = solver.cp
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA unavailable")
    except Exception:
        pytest.skip("CUDA unavailable")
    old = solver.core_fft.get_fft_backend_status()
    solver.core_fft.set_fft_backend("cupy")
    solver.core_fft.set_cupy_plan_mode("manual", preserve_cache=True)
    yield cp
    solver.core_fft.set_fft_backend(old["requested"])
    solver.core_fft.set_cupy_plan_mode(old["cupy_plan_mode"], preserve_cache=True)


def problem(**extra):
    shape = (6, 6, 6)
    return dict(
        phase_array=np.zeros(shape, np.uint8),
        ori_array=np.zeros(shape + (3,)),
        backend="cupy",
        dtype="float32",
        rtol=1e-5,
        initial_increment=0.25,
        return_fields=True,
        **extra,
    )


def physical_history(result):
    return [
        {k: v for k, v in row.items() if k != "seconds"} for row in result["history"]
    ]


def assert_controls(result):
    mask = np.asarray(result["configuration"]["loading"]["strain_control"])
    for row in result["history"]:
        e, s = np.asarray(row["strain"]), np.asarray(row["stress"])
        np.testing.assert_allclose(
            e[mask], np.asarray(row["target_strain_mandel"])[mask], atol=1e-7, rtol=1e-5
        )
        np.testing.assert_allclose(
            s[~mask], np.asarray(row["target_stress"])[~mask], atol=2e-7, rtol=1e-5
        )
        assert row["residual"] <= result["rtol"]
        assert row["macro_residual"] <= result["rtol"]
        assert row["macro_strain_residual"] <= result["rtol"]
        assert row["hill_mandel_error"] <= 5 * result["rtol"]
        assert row["plastic_dissipation"] >= 0


def test_interpolation_canonicalization_and_input_ownership():
    value = loading(
        mask=[True, False, False, False, False, True],
        strain=[
            [0.0] * 6,
            [0.001, 99.0, 99.0, 99.0, 99.0, 0.002],
            [0.001, 9.0, 9.0, 9.0, 9.0, -0.001],
        ],
        stress=[
            [0.0] * 6,
            [99.0, -0.01, -0.02, 0.001, 0.0, 99.0],
            [9.0, -0.03, -0.04, -0.001, 0.0, 9.0],
        ],
        parameter=[0.0, 2.0, 5.0],
    )
    before = copy.deepcopy(value)
    path = MixedLoadingPath.from_mapping(value)
    e, s = path.at(1.0)
    np.testing.assert_array_equal(e, [0.0005, 0.0, 0.0, 0.0, 0.0, 0.001])
    np.testing.assert_array_equal(s, [0.0, -0.005, -0.01, 0.0005, 0.0, 0.0])
    np.testing.assert_array_equal(path.at(2.0)[0], [0.001, 0.0, 0.0, 0.0, 0.0, 0.002])
    np.testing.assert_allclose(path.at(3.5)[0], [0.001, 0.0, 0.0, 0.0, 0.0, 0.0005])
    assert value == before
    e[0] = 8.0
    assert path.strain[1, 0] == 0.001
    for field in (path.strain_control, path.strain, path.stress, path.parameter):
        assert not field.flags.writeable
    for outside in (-0.01, 5.01, np.nan):
        with pytest.raises(ValueError, match="outside"):
            path.at(outside)


@pytest.mark.parametrize(
    "replacement",
    [
        dict(strain_control=[1, 0, 0, 0, 0, 0]),
        dict(strain_control=[True] * 5),
        dict(strain_control=["yes"] * 6),
        dict(strain=[[0.0] * 6]),
        dict(strain=[[0.0] * 5] * 2),
        dict(stress=[[0.0] * 6] * 3),
        dict(stress=[[np.nan] * 6] * 2),
        dict(parameter=[0.0, 0.0]),
        dict(parameter=[1.0, 0.0]),
        dict(parameter=[0.0, np.inf]),
        dict(parameter=[0.0, 0.5, 1.0]),
        dict(unrecognized=1),
    ],
)
def test_loading_validation(replacement):
    with pytest.raises(ValueError):
        MixedLoadingPath.from_mapping(loading() | replacement)


def test_default_parameter_and_multiaxial_clock():
    path = MixedLoadingPath.from_mapping(loading(mask=[False] * 6))
    np.testing.assert_array_equal(path.parameter, [0.0, 1.0])
    rows = [
        dict(
            target_parameter=0.0,
            target_strain_mandel=[0.0] * 6,
            target_stress=[0.0] * 6,
        ),
        dict(
            target_parameter=1.0,
            target_strain_mandel=[0.0] * 6,
            target_stress=[-0.01] * 3 + [0.0] * 3,
        ),
        dict(
            target_parameter=2.0,
            target_strain_mandel=[0.0] * 6,
            target_stress=[0.0] * 6,
        ),
    ]
    solver.annotate_pseudotime(rows, 0.01, 0.01, [False] * 6)
    np.testing.assert_allclose(
        [r["pseudo_time"] for r in rows], [0.0, np.sqrt(3.0), 2 * np.sqrt(3.0)]
    )
    assert [r["load_factor"] for r in rows] == [0.0, 1.0, 2.0]


@pytest.mark.gpu
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("route", ["classic", "pipeline"])
@pytest.mark.parametrize("model", ["j2", "drucker_prager"])
def test_elastic_analytic_general_controls(gpu, mask, route, model):
    # Supply mutually compatible subsets of one exact homogeneous solution;
    # no CPU equilibrium solve is used as the oracle.
    e = np.array([0.0008, -0.0002, -0.0001, 0.0003, -0.0001, 0.0002])
    E, nu = 3.0, 0.3
    mu, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    s = 2 * mu * e
    s[:3] += (K - 2 * mu / 3) * e[:3].sum()
    p = problem(
        model=model,
        integrator="backward_euler",
        coordinate_pcg=route,
        loading=loading(mask=mask, strain=[np.zeros(6), e], stress=[np.zeros(6), s]),
    )
    if model == "drucker_prager":
        p["matrix"] = dict(friction=0.3, dilatancy=0.1)
    result = solver.solve_nonlinear_homogenization(p)
    assert not result["rejected_steps"]
    assert_controls(result)
    np.testing.assert_allclose(result["history"][-1]["strain"], e, atol=1e-8, rtol=2e-5)
    np.testing.assert_allclose(result["history"][-1]["stress"], s, atol=1e-7, rtol=2e-5)
    assert not np.any(result["state"].equivalent)
    assert result["history"][-1]["target_parameter"] == 1.0


@pytest.mark.gpu
@pytest.mark.parametrize("mask", MASKS)
def test_pipeline_free_mean_counts_against_production_operator(gpu, mask):
    cp = gpu
    p = problem()
    material = DruckerPragerMaterial(friction=0.3, dilatancy=0.1)
    ids, table = solver._table(
        p["phase_array"],
        p["ori_array"],
        material,
        dict(E_L=70.0, E_T=8.0, nu_LT=0.25, nu_TT=0.3, G_LT=5.0),
    )
    ids, table = cp.asarray(ids), cp.asarray(table, dtype="float32")
    projector = solver.MixedProjector(
        ids.shape, (1.0, 1.0, 1.0), mask, cp, spectral_fusion=True
    )
    engine = TangentPipeline(projector, material, ids, table)
    constitutive = CuPyDruckerPrager(cp, material, ids, table, "float32")
    rng = np.random.default_rng(294)
    response = constitutive.update(
        cp.asarray(rng.normal(0.0, 0.004, (6,) + ids.shape), dtype="float32"),
        initial_state(ids.shape, cp, cp.float32),
    )
    pipeline = engine.pipeline
    try:
        pipeline.bind(
            response.a,
            response.b,
            response.direction,
            response.bulk,
            response.coupling_to_direction,
            response.coupling_to_pressure,
        )
        z = engine.Q.adjoint(
            cp.asarray(rng.normal(size=(6,) + ids.shape), dtype="float32")
        )
        fused = pipeline.to_fused(z)
        cp.testing.assert_allclose(
            pipeline.to_production(fused), z, rtol=2e-6, atol=2e-6
        )
        expected = engine.Q.adjoint(constitutive.tangent(engine.Q.strain(z), response))
        actual = pipeline.to_production(pipeline.matvec(fused))
        cp.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)
        # The production projector must retain exactly the chosen free means.
        constant = cp.ones((6,) + ids.shape, dtype="float32")
        expected_mean = (~np.asarray(mask)).astype(float)
        cp.testing.assert_allclose(
            cp.mean(projector(constant), axis=(1, 2, 3)),
            cp.asarray(expected_mean),
            atol=2e-6,
        )
    finally:
        pipeline.release()


def confined_nonproportional():
    return loading(
        mask=[True, False, False, False, False, True],
        strain=[
            [0.0] * 6,
            [0.0] * 6,
            [0.008, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.0, 0.0, 0.0, 0.0, 0.004],
        ],
        stress=[[0.0] * 6] + [[0.0, -0.005, -0.005, 0.0, 0.0, 0.0]] * 3,
        parameter=[0.0, 0.25, 0.5, 1.0],
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_confinement_nonproportional_unloading_and_routes(gpu, dilatancy):
    p = problem(
        loading=confined_nonproportional(),
        model="drucker_prager",
        matrix=dict(friction=0.3, dilatancy=dilatancy),
    )
    p.update(initial_increment=0.125)
    classic = solver.solve_nonlinear_homogenization(p | dict(coordinate_pcg="classic"))
    pipeline = solver.solve_nonlinear_homogenization(
        p | dict(coordinate_pcg="pipeline")
    )
    for result in (classic, pipeline):
        assert_controls(result)
        assert not result["rejected_steps"]
        assert result["history"][-1]["equivalent_plastic_mean"] > 0
        assert np.all(
            np.diff([row["target_parameter"] for row in result["history"]]) > 0
        )
        assert np.all(np.diff([row["pseudo_time"] for row in result["history"]]) > 0)
        for knot in (0.0, 0.25, 0.5, 1.0):
            assert knot in [row["target_parameter"] for row in result["history"]]
    for field in ("strain", "stress"):
        np.testing.assert_allclose(
            pipeline["fields"][field], classic["fields"][field], rtol=1e-4, atol=2e-7
        )


@pytest.mark.gpu
def test_scalar_legacy_matches_general_path(gpu):
    path = [0.0, 0.002, 0.001]
    p = problem(integrator="backward_euler", predictor=False, elastic_segments=False)
    p.update(initial_increment=0.00025)
    scalar = solver.solve_nonlinear_homogenization(p | dict(path=path))
    general_loading = loading(
        strain=[[e, 0.0, 0.0, 0.0, 0.0, 0.0] for e in path],
        stress=[[0.0] * 6] * 3,
        parameter=[0.0, 0.002, 0.003],
    )
    general = solver.solve_nonlinear_homogenization(p | dict(loading=general_loading))
    for key in ("strain", "stress"):
        np.testing.assert_allclose(
            [r[key] for r in general["history"]],
            [r[key] for r in scalar["history"]],
            atol=1e-7,
            rtol=2e-5,
        )
    np.testing.assert_allclose(
        [r["pseudo_time"] for r in general["history"]],
        [r["pseudo_time"] for r in scalar["history"]],
        atol=1e-12,
    )
    assert "loading" not in scalar["configuration"]


@pytest.mark.gpu
def test_restart_preserves_state_and_checks_boundary_conditions(gpu, tmp_path):
    value = confined_nonproportional()
    first_path = {
        key: (item[:3] if key != "strain_control" else item)
        for key, item in value.items()
    }
    continuation = {
        key: (item[2:] if key != "strain_control" else item)
        for key, item in value.items()
    }
    checkpoint = tmp_path / "confirmed.npz"
    p = problem(
        model="drucker_prager",
        matrix=dict(friction=0.3, dilatancy=0.1),
        predictor=False,
    )
    p.update(initial_increment=0.125)
    full = solver.solve_nonlinear_homogenization(p | dict(loading=value))
    first = solver.solve_nonlinear_homogenization(
        p | dict(loading=first_path, checkpoint_path=checkpoint)
    )
    resumed = solver.solve_nonlinear_homogenization(
        p | dict(loading=continuation, restart_from=checkpoint)
    )
    assert physical_history(resumed)[: len(first["history"])] == physical_history(first)
    for key in ("strain", "stress"):
        np.testing.assert_allclose(
            resumed["fields"][key], full["fields"][key], atol=2e-7, rtol=1e-4
        )
    np.testing.assert_allclose(
        resumed["state"].plastic_strain,
        full["state"].plastic_strain,
        atol=2e-7,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        resumed["state"].equivalent, full["state"].equivalent, atol=2e-7, rtol=1e-4
    )
    np.testing.assert_allclose(
        [r["pseudo_time"] for r in resumed["history"]],
        [r["pseudo_time"] for r in full["history"]],
        atol=1e-12,
    )
    with np.load(checkpoint) as ck:
        metadata = json.loads(str(ck["metadata"]))
    assert metadata["version"] == 2
    assert metadata["loading_state"]["parameter"] == 0.5
    changes = [
        dict(strain_control=[False] * 6),
        dict(parameter=[0.6, 1.0]),
        dict(stress=[[0.0, -0.004, -0.005, 0.0, 0.0, 0.0], continuation["stress"][-1]]),
    ]
    for change in changes:
        with pytest.raises(ValueError, match="mismatch|checkpoint"):
            solver.solve_nonlinear_homogenization(
                p | dict(loading=continuation | change, restart_from=checkpoint)
            )


@pytest.mark.gpu
def test_rejected_attempt_keeps_zero_confirmed_checkpoint(gpu, tmp_path, monkeypatch):
    checkpoint = tmp_path / "rejected.npz"
    original = TangentPipeline.solve

    def fail(self, *args, **kwargs):
        raise solver.NonlinearConvergenceError("injected mixed tangent failure")

    monkeypatch.setattr(TangentPipeline, "solve", fail)
    p = problem(loading=loading(), globalization="newton", checkpoint_path=checkpoint)
    p.update(initial_increment=1.0, min_increment=1.0)
    with pytest.raises(solver.NonlinearConvergenceError, match="minimum increment"):
        solver.solve_nonlinear_homogenization(p)
    with np.load(checkpoint) as ck:
        assert (
            not np.any(ck["strain"])
            and not np.any(ck["plastic_strain"])
            and not np.any(ck["equivalent"])
        )
        metadata = json.loads(str(ck["metadata"]))
    assert metadata["loading_state"]["parameter"] == 0.0
    assert len(metadata["history"]) == 1
    monkeypatch.setattr(TangentPipeline, "solve", original)
    result = solver.solve_nonlinear_homogenization(p | dict(restart_from=checkpoint))
    assert_controls(result)


@pytest.mark.gpu
def test_general_exports_and_arc_pseudosteps(gpu, tmp_path):
    p = problem(loading=loading(), output_dir=tmp_path)
    p.pop("initial_increment")
    p["initial_pseudo_step"] = 0.025
    result = solver.solve_nonlinear_homogenization(p)
    assert_controls(result)
    np.testing.assert_allclose(
        [r["pseudo_time"] for r in result["history"]],
        [0.0, 0.025, 0.05, 0.075, 0.1],
        atol=1e-12,
    )
    with (tmp_path / "curve.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert "target_parameter" in rows[0] and "target_stress_Mandel_GPa_22" in rows[0]
    assert float(rows[-1]["target_strain_Mandel_11"]) == 0.001
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["loading"]["kind"] == "piecewise_linear_mixed"
    assert metadata["loading"]["other_mean_stresses"] != "zero"
    assert (tmp_path / "curve.png").exists() and (
        tmp_path / "constitutive.csv"
    ).exists()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "options",
    [
        dict(path=[0.0, 0.01]),
        dict(elastic_segments=True),
        dict(pseudo_time_stress_scale_GPa=0.0),
    ],
)
def test_general_options_reject_ambiguous_or_unsupported_contract(gpu, options):
    with pytest.raises(ValueError):
        solver.solve_nonlinear_homogenization(problem(loading=loading()) | options)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "value",
    [
        loading(
            strain=[[0.001, 0.0, 0.0, 0.0, 0.0, 0.0], [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        loading(
            stress=[
                [0.0, -0.001, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    ],
)
def test_nonzero_initial_controls_rejected_clearly(gpu, value):
    with pytest.raises(ValueError, match="begin with zero prescribed"):
        solver.solve_nonlinear_homogenization(problem(loading=value))
