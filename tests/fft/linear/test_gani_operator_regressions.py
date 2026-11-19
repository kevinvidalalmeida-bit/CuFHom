"""CPU regressions; CUDA checks require an explicitly serialized GPU slot.

CPU: python -B -m pytest -p no:cacheprovider -q <this file> -k 'not cuda'
CUDA: CUFHOM_SECTION3_GANI_GPU_TESTS=1 python -B -m pytest -q <this file> -k cuda
"""

import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _CPUArray(np.ndarray):
    """Only the device-to-host copy API used by host solution retention."""

    def get(self, out):
        np.copyto(out, self)


def _cpu_array(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(_CPUArray)


@pytest.fixture
def cpu_arrays():
    # No CUDA runtime, allocation, compilation or device discovery is called.
    return SimpleNamespace(
        asarray=_cpu_array,
        asnumpy=np.asarray,
        sum=np.sum,
        empty=lambda shape, dtype: np.empty(shape, dtype).view(_CPUArray),
        zeros=lambda shape, dtype: np.zeros(shape, dtype).view(_CPUArray),
        float32=np.float32,
        float64=np.float64,
        complex64=np.complex64,
        complex128=np.complex128,
        uint16=np.uint16,
        uint32=np.uint32,
        int32=np.int32,
        cuda=SimpleNamespace(Stream=lambda **kw: object()),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (80, 225, 128),
        (84, 225, 128),
        (80, 375, 256),
        (32, 32, 32),
        (128, 128, 128),
        (24, 30, 40),
    ],
)
def test_cpu_real_evaluation_config_covers_every_material_line(
    monkeypatch, cpu_arrays, shape
):
    from cufhom.fft.linear import evaluation, fused

    limit = 99 * 1024

    def properties():
        return {"sharedMemPerBlockOptin": limit}

    monkeypatch.setattr(fused, "cp", object())
    monkeypatch.setattr(fused, "device_properties", properties)
    monkeypatch.setattr(evaluation, "device_properties", properties)
    monkeypatch.setattr(evaluation, "cp", cpu_arrays)
    monkeypatch.setattr(evaluation, "_PARTS", {})
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __call__(self, grid, block, args, **kwargs):
            if self.name == "kc_eval":
                calls.append((int(args[-3]), grid[0]))

    monkeypatch.setattr(
        evaluation,
        "_compile_many",
        lambda specs: [Kernel(name) for source, name, shared in specs],
    )
    working = fused.default_config(shape, limit, np.float32)
    original = fused.default_config(shape, limit, np.float64)
    Q = SimpleNamespace(
        shape=shape,
        tile=working["ta"],
        norm=1 / np.sqrt(np.prod(shape)),
        P=SimpleNamespace(lengths=(0.8, 1.3, 1.1)),
        transverse=0.5,
        longitudinal=0.3,
    )
    descriptor = SimpleNamespace(material_table=np.zeros((2, 21)))
    assert evaluation.evaluation_supported(Q, descriptor)
    if shape[1] == 225:
        assert working["slabs"] == 12 and working["lpb"] == 4
        assert original["slabs"] == 6 and original["lpb"] == 4
        assert shape[1] * original["slabs"] % original["lpb"] != 0
    if shape[1] == 375:
        assert working["slabs"] == 4 and original["slabs"] == 2
        assert working["lpb"] == original["lpb"] == 4

    pipeline = evaluation.EvaluationPipeline(
        Q, descriptor, "float64", ids=np.zeros(1, np.uint16)
    )
    cfg = pipeline.config
    if shape[1] * original["slabs"] % original["lpb"] == 0:
        assert cfg == original  # Usual configurations keep their exact tuning.
    assert cfg["lpb"] == original["lpb"]
    assert pipeline.grid_c == -(-shape[0] * shape[1] // original["lpb"])
    assert pipeline.S is None  # Configuration/launch inspection needs no spectrum.
    assert max(pipeline.smem.values()) <= limit
    assert pipeline.smem["material"] == (shape[2] // 2) * (6 * original["lpb"] + 2) * 16
    pipeline.partial.fill(0)
    pipeline.mean.fill(0)
    pipeline.forward(np.zeros(6), None)

    covered = []
    for (group0, groups), start in zip(
        calls, range(0, shape[0], cfg["slabs"]), strict=True
    ):
        stop = min(shape[0], start + cfg["slabs"])
        first, last = group0 * cfg["lpb"], (group0 + groups) * cfg["lpb"]
        # Both complete slabs and the tail must exactly match the FFT slab.
        assert (first, last) == (start * shape[1], stop * shape[1])
        covered.extend(range(first, last))
    np.testing.assert_array_equal(covered, np.arange(shape[0] * shape[1]))


@pytest.fixture
def cpu_homogeneous_load(monkeypatch, cpu_arrays):
    """Run actual mixed-load acceptance/assembly with NumPy homogeneous FFTs.

    Only CUDA services and the known-zero compatible projection are replaced;
    _HostVector, mixed_load and _six_mixed_loads retain their real control flow.
    Poisoned pinned allocations make an unmaterialized logical zero deterministic.
    """
    from cufhom.fft.linear import coordinates

    active_capacity = ModuleType("cufhom.fft.linear.active_capacity")
    monkeypatch.setitem(sys.modules, active_capacity.__name__, active_capacity)

    shape = (8, 8, 8)
    stiffness = 2 * np.eye(6)
    stiffness[:3, :3] += 1.0
    pinned = []
    cupyx = ModuleType("cupyx")

    def empty_pinned(shape, dtype):
        value = np.full(shape, complex(np.nan, np.nan), dtype=dtype)
        pinned.append(value)
        return value

    cupyx.empty_pinned = empty_pinned
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setattr(coordinates, "cp", cpu_arrays)

    class Coordinates:
        def __init__(self, *args):
            self.shape, self.size, self.tile = shape, 3 * np.prod(shape) // 2, None

        def tiled(self, tile):
            self.tile = tile
            return self

    class HomogeneousOperator:
        def __init__(self, *args):
            pass

        def work(self):
            real = np.empty((6,) + shape)
            spectrum = np.empty(
                (6,) + shape[:-1] + (shape[-1] // 2 + 1,), np.complex128
            )
            return real, spectrum, real

        def constitutive(self, work):
            work[...] = np.einsum("ij,j...->i...", stiffness, work)

        def forward(self, real, spectrum):
            spectrum[...] = np.fft.rfftn(real, axes=(1, 2, 3))

    def residual_map(Q, spectrum, residual, audit):
        # A homogeneous material under uniform strain has only a mean stress.
        np.testing.assert_array_equal(spectrum.reshape(6, -1)[:, 1:], 0.0)
        residual.fill(0)
        return 0.0

    def mixed_dot(grid, block, args):
        a, b, partial, count = args
        partial.fill(0)
        partial[0] = np.vdot(a, b).real

    def real_dot(a, b):
        assert np.isfinite(a).all(), "accepted coordinates read before initialization"
        return float(np.vdot(a, b).real)

    def unexpected(*args, **kwargs):
        pytest.fail("a homogeneous accepted load must not iterate or fold")

    pipeline = SimpleNamespace(
        config={"ta": 1}, release=lambda: None, settle=unexpected, prefix=unexpected
    )
    monkeypatch.setattr(coordinates, "MixedProjector", lambda *args: None)
    monkeypatch.setattr(coordinates, "SymmetricCoordinates", Coordinates)
    monkeypatch.setattr(
        coordinates, "_working_descriptor", lambda pb, descriptor: descriptor
    )
    monkeypatch.setattr(coordinates, "_solve_pipeline", lambda *args: pipeline)
    monkeypatch.setattr(coordinates, "_evaluation_pipeline", lambda *args: None)
    monkeypatch.setattr(coordinates, "_phase_spectra", lambda *args: None)
    monkeypatch.setattr(
        coordinates,
        "_material_constants",
        lambda *args: dict(voigt=stiffness, bounds={"gamma_minus": 1.0}),
    )
    monkeypatch.setattr(coordinates, "_residual_map", residual_map)
    monkeypatch.setattr(
        coordinates, "reduction_blocks", lambda n: max(1, (n + 255) // 256)
    )
    monkeypatch.setattr(coordinates, "_mixed_kernel", lambda name: mixed_dot)
    monkeypatch.setattr(coordinates, "_real_dot", real_dot)
    monkeypatch.setattr(coordinates, "fused_cg", unexpected)
    monkeypatch.setattr(coordinates._HostVector, "fold", unexpected)
    active_capacity.ActiveCapacity = HomogeneousOperator
    pb = SimpleNamespace(Y=(0.8, 1.3, 1.1), solver={})
    return coordinates, shape, stiffness, pinned, pb


@pytest.mark.parametrize("energy", [None, 1e-8])
def test_cpu_homogeneous_host_zero_accepted_without_fold(cpu_homogeneous_load, energy):
    coordinates, shape, stiffness, pinned, pb = cpu_homogeneous_load
    result = coordinates._six_mixed_loads(
        None,
        None,
        pb,
        5 / 3,
        1.0,
        shape,
        dict(rtol=1e-6, maxiter=100),
        energy,
        "host",
        "df64",
    )
    assert result["retained"] == "host"
    assert all(info["kit"] == 0 for info in result["infos"])
    np.testing.assert_array_equal(result["C"], stiffness)
    np.testing.assert_array_equal(result["energy_correction"], 0.0)
    assert len(pinned) == 11  # Six accumulated fields, five retained FP32 copies.
    for field in pinned:
        np.testing.assert_array_equal(field, 0.0)


def test_cpu_host_allocation_stays_lazy_for_iterating_loads(cpu_homogeneous_load):
    coordinates, shape, stiffness, pinned, pb = cpu_homogeneous_load
    Q = SimpleNamespace(shape=shape, size=3 * np.prod(shape) // 2)
    solution = coordinates._HostVector.coordinates(Q)
    assert solution.zero
    assert np.isnan(solution.host).all()  # No unconditional host zero-fill.
    assert len(solution.slabs) == 2
    assert sum(slab.nbytes for slab in solution.slabs) <= 2 * coordinates._SLAB_BYTES


cuda_only = pytest.mark.skipif(
    os.environ.get("CUFHOM_SECTION3_GANI_GPU_TESTS") != "1",
    reason="requires explicit serialized CUDA slot",
)


@pytest.mark.gpu
@cuda_only
@pytest.mark.parametrize("energy", [None, 1e-8])
def test_cuda_homogeneous_host_zero_accepted_without_fold(monkeypatch, energy):
    cp = pytest.importorskip("cupy")
    import cupyx
    from cufhom.fft.linear import coordinates
    from cufhom.fft.linear.common import IndexedMaterial

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    empty_pinned = cupyx.empty_pinned

    def poisoned(shape, dtype):
        value = empty_pinned(shape, dtype)
        value.fill(complex(np.nan, np.nan))
        return value

    def unexpected(*args, **kwargs):
        pytest.fail("homogeneous zero fluctuation must be accepted without fold")

    monkeypatch.setattr(cupyx, "empty_pinned", poisoned)
    monkeypatch.setattr(coordinates._HostVector, "fold", unexpected)
    stiffness = 2 * np.eye(6)
    stiffness[:3, :3] += 1.0
    material = IndexedMaterial(
        np.zeros((8, 8, 8), np.int32), stiffness[None], dtype="float64"
    )
    result = coordinates.homogenize_gani_mixed(
        material, (0.8, 1.3, 1.1), stiffness, residency="host", energy=energy
    )
    cp.cuda.Device().synchronize()
    assert all(info["kit"] == 0 for info in result["infos"])
    np.testing.assert_allclose(result["C"], stiffness, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result["energy_correction"], 0.0)


@pytest.mark.gpu
@cuda_only
@pytest.mark.parametrize("arithmetic", ["df64", "float64"])
def test_cuda_aligned_evaluation_matches_library(monkeypatch, arithmetic):
    cp = pytest.importorskip("cupy")
    from cufhom.fft.linear import coordinates, evaluation, fused
    from cufhom.fft.linear.active_capacity import ActiveCapacity
    from cufhom_core.tensors import fft

    from tests.fft.linear.test_linear_fused_fft import _problem

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    previous = fft.get_fft_backend()
    fft.set_fft_backend("cupy")
    try:
        shape = (80, 225, 128)
        Q, descriptor, rng = _problem(shape, seed=20261023)
        working = fused.FusedPipeline(Q, descriptor)
        Q = Q.tiled(working.config["ta"])
        descriptor.material_table = descriptor.material_table.astype(cp.float64)
        evaluator = evaluation.EvaluationPipeline(
            Q, descriptor, arithmetic, ids=working.ids
        )
        z = Q.adjoint(cp.asarray(rng.standard_normal((6,) + shape)))
        macro = np.array([1.0, 0.0, 0.0, 0.25, 0.0, 0.0])
        residual = cp.empty(Q.size, cp.complex64)
        evaluator.inverse(z)
        rr, audit, mean = evaluator.forward(macro, residual)
        operator = ActiveCapacity(descriptor, shape, Q.P.lengths, Q)
        work, spectrum, real = operator.work()
        operator.map(z, spectrum, False)
        operator.inverse(spectrum, real)
        real += cp.asarray(macro)[:, None, None, None]
        operator.constitutive(work)
        operator.forward(real, spectrum)
        reference = cp.empty_like(residual)
        reference_audit = coordinates._residual_map(Q, spectrum, reference, True)
        cp.testing.assert_allclose(residual, reference, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(audit, reference_audit, rtol=2e-9)
        np.testing.assert_allclose(
            rr, float(cp.vdot(reference, reference).real), rtol=3e-6
        )
        np.testing.assert_allclose(
            mean, cp.asnumpy(spectrum[:, 0, 0, 0].real), rtol=2e-9
        )
    finally:
        fft.set_fft_backend(previous)
