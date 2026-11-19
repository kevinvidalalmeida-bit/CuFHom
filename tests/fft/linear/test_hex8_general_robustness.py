"""CPU-only validation, FP64 reductions, and HEX8 output diagnostics."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from cufhom.fft.linear import hex8_general as hex8


class NoCastArray(np.ndarray):
    """Fail if a reduction tries to promote a whole retained field."""

    def astype(self, *args, **kwargs):
        raise AssertionError("Whole-vector dtype conversion during reduction")


@pytest.fixture
def operator():
    labels = np.indices((2, 3, 2)).sum(0) % 2
    rng = np.random.default_rng(58)
    a = rng.normal(size=(2, 6, 6))
    stiffness = a.transpose(0, 2, 1) @ a + 3 * np.eye(6)
    return hex8.Hex8Operator(labels, stiffness, backend="numpy")


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.float64, np.float64),
    ],
)
@pytest.mark.parametrize(
    "left,right",
    [
        ([1e20, 1e20], [1e20, 1e20]),  # Multiplication must not overflow in FP32.
        ([2**24, 1, 1, 1], [1, 1, 1, 1]),  # Accumulation must retain small terms.
    ],
)
def test_dot_accumulates_products_in_fp64_without_vector_casts(
    operator, a_dtype, b_dtype, left, right
):
    a = np.array(left, dtype=a_dtype).view(NoCastArray)
    b = np.array(right, dtype=b_dtype).view(NoCastArray)
    expected = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    with np.errstate(over="raise", invalid="raise"):
        assert operator.dot(a, b) == expected


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_complex_stiffness_is_rejected_before_casting(packed, imaginary):
    c = np.eye(6, dtype=complex)[None]
    c[0, 0, 0] += imaginary * 1j
    if packed:
        c = np.stack([c[:, i, j] for i, j in hex8._PAIRS], axis=1)
    with pytest.raises(ValueError, match="real"):
        hex8.Hex8Operator(np.zeros((2, 2, 2), int), c, backend="numpy")


def test_complex_reference_is_rejected():
    with pytest.raises(ValueError, match="real"):
        hex8.Hex8Operator(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            reference=np.eye(6, dtype=complex),
            backend="numpy",
        )


def test_material_overflow_in_requested_precision_is_rejected():
    with pytest.raises(ValueError, match="remain finite"):
        hex8.Hex8Operator(
            np.zeros((2, 2, 2), int),
            (np.eye(6) * 1e40)[None],
            dtype="float32",
            backend="numpy",
        )


def test_numpy_inverse_nonfinite_output_is_rejected(monkeypatch):
    def nonfinite_inverse(symbol):
        inverse = np.zeros_like(symbol)
        inverse[1, 0, 0, 0, 0] = np.inf
        return inverse

    monkeypatch.setattr(np.linalg, "inv", nonfinite_inverse)
    with pytest.raises(ValueError, match="Reference HEX8 inverse"):
        hex8.Hex8Operator(np.zeros((2, 2, 2), int), np.eye(6)[None], backend="numpy")


@pytest.mark.parametrize(
    "parameters",
    [
        {"rtol": -1.0},
        {"atol": -1.0},
        {"rtol": 0.0, "atol": 0.0},
        {"rtol": np.nan},
        {"rtol": np.inf},
        {"atol": np.nan},
        {"atol": np.inf},
        {"rtol": 1j},
        {"atol": 1e-6 + 0j},
        {"rtol": [1e-6]},
        {"atol": [0.0]},
        {"rtol": "1e-6"},
        {"atol": None},
        {"rtol": True},
        {"maxiter": 0},
        {"maxiter": -1},
        {"maxiter": 1.5},
        {"maxiter": np.nan},
        {"maxiter": np.inf},
        {"maxiter": -np.inf},
        {"maxiter": 2 + 0j},
        {"maxiter": [2]},
        {"maxiter": "2"},
        {"maxiter": None},
        {"maxiter": True},
    ],
)
def test_invalid_solver_parameters_fail_before_operator_work(monkeypatch, parameters):
    def unexpected(*args, **kwargs):
        raise AssertionError("Operator work happened before parameter validation")

    monkeypatch.setattr(hex8, "Hex8Operator", unexpected)
    with pytest.raises(ValueError, match="tolerances"):
        hex8.solve_hex8_load(
            SimpleNamespace(rhs=unexpected), np.eye(6)[0], **parameters
        )
    with pytest.raises(ValueError, match="tolerances"):
        hex8.homogenize_hex8(None, None, backend="numpy", **parameters)


@pytest.mark.parametrize("budget", [np.int64(3000), 3000.0, np.array(3000)])
def test_integral_budgets_are_used_for_nonzero_loads(operator, budget):
    info = hex8.solve_hex8_load(operator, np.eye(6)[0], maxiter=budget, rtol=1e-10)
    assert info["iterations"] > 0
    assert info["accepted"]
    assert info["residual_inf"] <= info["threshold"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"rtol": 0.0, "atol": 1e-10},
        {"rtol": 1e308, "atol": 1e308},
    ],
)
def test_finite_tolerances_are_validated_individually(operator, parameters):
    info = hex8.solve_hex8_load(operator, np.zeros(6), **parameters)
    assert info["iterations"] == 0
    assert info["accepted"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_loading_cannot_pass_zero_iteration_acceptance(
    operator, monkeypatch, value
):
    rhs = np.zeros((3, *operator.shape))
    rhs.flat[0] = value
    monkeypatch.setattr(operator, "rhs", lambda macro: rhs)
    with pytest.raises(RuntimeError, match="loading is nonfinite"):
        hex8.solve_hex8_load(operator, np.eye(6)[0])


@pytest.mark.parametrize("dtype,rtol", [("float32", 2e-5), ("float64", 1e-10)])
def test_cpu_outputs_are_finite_symmetric_and_physically_accepted(
    operator, dtype, rtol
):
    result = hex8.homogenize_hex8(
        operator.index,
        operator.C,
        backend="numpy",
        dtype=dtype,
        rtol=rtol,
        return_fields=True,
    )
    audit = hex8.Hex8Operator(operator.index, operator.C, backend="numpy", dtype=dtype)
    for key in ("C", "C_stress", "energy_correction", "reference", "symmetry_defect"):
        assert np.isfinite(result[key]).all()
    np.testing.assert_array_equal(result["C"], result["C"].T)
    assert np.linalg.eigvalsh(result["C"])[0] > 0
    assert result["symmetry_defect"] < 1e-12
    for macro, u, info in zip(np.eye(6), result["solutions"], result["infos"]):
        assert np.isfinite(u).all()
        assert info["accepted"]
        fresh = audit.rhs(macro) - audit.action(u, audit=True)
        assert np.max(np.abs(fresh)) <= info["threshold"]


def mock_loads(monkeypatch, stress, solutions=None, residuals=None):
    shape = (3, 2, 2, 2)
    if solutions is None:
        solutions = [np.zeros(shape, np.float32).view(NoCastArray) for _ in range(6)]
    if residuals is None:
        residuals = [np.zeros(shape, np.float64) for _ in range(6)]

    def solve(operator, macro, **kwargs):
        a = int(np.argmax(macro))
        return dict(
            solution=solutions[a],
            residual=residuals[a],
            accepted=True,
            energy_error_upper=0.0,
            energy_error_lower=0.0,
        )

    monkeypatch.setattr(hex8, "solve_hex8_load", solve)
    monkeypatch.setattr(
        hex8.Hex8Operator, "mean_stress", lambda self, u, macro: stress @ macro
    )


def homogenize_mock_loads(**kwargs):
    return hex8.homogenize_hex8(
        np.zeros((2, 2, 2), int),
        np.eye(6)[None],
        backend="numpy",
        dtype="float32",
        **kwargs,
    )


def expected_energy(stress, correction):
    """C[a,b] = C_stress[b,a] - u_b.r_a for b <= a, mirrored."""
    lower = np.tril(stress.T - correction)
    return lower + np.tril(lower, -1).T


def test_host_energy_reduction_and_symmetry_diagnostic(monkeypatch):
    rng = np.random.default_rng(72)
    solutions = [
        rng.normal(size=(3, 2, 2, 2)).astype(np.float32).view(NoCastArray)
        for _ in range(6)
    ]
    residuals = [rng.normal(size=u.shape) * 1e-5 for u in solutions]
    # Earlier solution b with later residual a: no retained residuals needed.
    correction = np.array(
        [
            [
                math.fsum(
                    float(x) * float(y)
                    for x, y in zip(
                        solutions[min(a, b)].flat, residuals[max(a, b)].flat
                    )
                )
                for b in range(6)
            ]
            for a in range(6)
        ]
    )
    skew = np.zeros((6, 6))
    skew[0, 1], skew[1, 0] = 0.25, -0.25
    stress = 4 * np.eye(6) + skew + correction
    mock_loads(monkeypatch, stress, solutions, residuals)
    result = homogenize_mock_loads(storage="host", return_fields=True)
    np.testing.assert_allclose(
        result["energy_correction"], correction, atol=1e-19, rtol=1e-14
    )
    np.testing.assert_allclose(result["C_stress"], stress, atol=1e-15)
    np.testing.assert_allclose(
        result["C"], expected_energy(stress, correction), atol=1e-15
    )
    np.testing.assert_allclose(
        result["C"], 4 * np.eye(6) + np.tril(skew.T) + np.tril(skew.T, -1).T, atol=1e-15
    )
    np.testing.assert_array_equal(result["C"], result["C"].T)
    assert result["symmetry_defect"] == pytest.approx(
        np.linalg.norm(2 * skew) / np.linalg.norm(result["C"])
    )
    assert all(np.shares_memory(u, v) for u, v in zip(result["solutions"], solutions))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_symmetry_diagnostic_survives_norm_underflow_and_overflow(monkeypatch, scale):
    stress = np.eye(6)
    stress[0, 1], stress[1, 0] = 0.25, -0.25
    mock_loads(monkeypatch, scale * stress)
    result = homogenize_mock_loads()
    expected = expected_energy(stress, np.zeros((6, 6)))
    np.testing.assert_allclose(result["C"] / scale, expected, atol=1e-15)
    assert result["symmetry_defect"] == pytest.approx(
        np.sqrt(2 * 0.5**2) / np.linalg.norm(expected)
    )


@pytest.mark.parametrize("source", ["stress", "correction"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_assembled_outputs_are_rejected(monkeypatch, source, value):
    stress = np.eye(6)
    residuals = None
    if source == "stress":
        stress[0, 0] = value
    else:
        residuals = [np.zeros((3, 2, 2, 2)) for _ in range(6)]
        residuals[0].flat[0] = value
    mock_loads(monkeypatch, stress, residuals=residuals)
    with np.errstate(invalid="ignore"):
        with pytest.raises(RuntimeError, match="nonfinite"):
            homogenize_mock_loads()


def test_indefinite_effective_energy_is_rejected(monkeypatch):
    stress = np.eye(6)
    stress[0, 0] = -1
    mock_loads(monkeypatch, stress)
    with pytest.raises(RuntimeError, match="positive definite"):
        homogenize_mock_loads()
