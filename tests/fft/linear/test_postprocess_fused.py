"""Full effective-matrix products versus independent CPU64 contraction."""

import numpy as np
import pytest
from cufhom_core.postprocess import assembly_matrix_fused
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_general_real_fields(backend, dtype, noncontiguous):
    if backend == "cupy" and cp is None:
        pytest.skip("CuPy unavailable")
    xp = np if backend == "numpy" else cp
    rng = np.random.default_rng(309)
    shape, dimension = (5, 4, 7), 3
    fields = []
    for i in range(dimension):
        data = xp.asarray(rng.normal(size=(dimension,) + shape).astype(dtype))
        if noncontiguous:
            data = data[..., ::-1]
        fields.append(
            Tensor(name=str(i), val=data, order=1, N=shape, Fourier=False, fft_form="r")
        )
    # Deliberately non-symmetric map: test all entries independently instead
    # of mirroring a triangle. It need not represent an elastic material.
    matrix = xp.asarray(rng.normal(size=(dimension, dimension)).astype(dtype))

    class Operator:
        N = shape

        def __call__(self, field):
            value = (matrix @ field.val.reshape(dimension, -1)).reshape(
                (dimension,) + shape
            )
            return field.copy(val=value)

    action = Operator()
    to_cpu = np.asarray if xp is np else cp.asnumpy
    before = [to_cpu(f.val).copy() for f in fields]
    expected = np.array(
        [
            [
                np.sum(to_cpu(action(a).val).astype(np.float64) * b.astype(np.float64))
                / np.prod(shape)
                for b in before
            ]
            for a in fields
        ]
    )
    candidate = assembly_matrix_fused(action, fields)
    tol = (
        2e-13
        if (dtype == np.float64 or (backend == "cupy" and not noncontiguous))
        else 5e-7
    )
    np.testing.assert_allclose(candidate, expected, rtol=tol, atol=tol)
    for original, field in zip(before, fields):
        np.testing.assert_array_equal(to_cpu(field.val), original)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(5, 4, 7), (8, 10, 12), (3, 7, 11)])
def test_indexed_assembly_all_entries_without_stress_fields(shape, monkeypatch):
    if cp is None:
        pytest.skip("CuPy unavailable")
    rng = np.random.default_rng(3811)
    matrices = rng.normal(size=(4, 6, 6))
    matrices = np.einsum("mki,mkj->mij", matrices, matrices) + np.eye(6)
    ids = rng.integers(0, 4, shape, dtype=np.int32)
    i, j = np.triu_indices(6)
    table = matrices[:, i, j].astype(np.float32)
    A = Tensor(
        name="indexed",
        val=cp.asarray(ids)[None, ...],
        order=1,
        N=shape,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    A.material_table = cp.ascontiguousarray(cp.asarray(table))
    arrays = rng.normal(size=(6, 6) + shape).astype(np.float32)
    fields = [
        Tensor(
            name=str(k),
            val=cp.asarray(data),
            order=1,
            N=shape,
            Fourier=False,
            fft_form="r",
        )
        for k, data in enumerate(arrays)
    ]
    # Independently reconstruct all 36 material coefficients on the CPU.
    rounded = np.zeros((4, 6, 6), np.float64)
    rounded[:, i, j] = rounded[:, j, i] = table
    stress64 = np.einsum("...ij,aj...->ai...", rounded[ids], arrays.astype(np.float64))
    expected = np.einsum(
        "aixyz,bixyz->ab", stress64, arrays.astype(np.float64)
    ) / np.prod(shape)
    from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction

    reduction = CupyWarpCGReduction(fields[0], max_blocks=16384)
    original = []
    for field in fields:
        stress = A(field)
        row = []
        for other in fields:
            reduction.dot_kernel(
                (reduction.blocks,),
                (reduction.threads,),
                (stress.val, other.val, reduction.partial, np.int64(reduction.total)),
            )
            row.append(
                float(cp.sum(reduction.partial, dtype=cp.float64)) / np.prod(shape)
            )
        original.append(row)

    # Fusion must bypass material-result allocations, not just reorder calls.
    def no_material_buffer(*args, **kwargs):
        raise AssertionError("Fused indexed assembly materialized a stress field")

    monkeypatch.setattr(Tensor, "__call__", no_material_buffer)
    actual = assembly_matrix_fused(A, fields)
    np.testing.assert_allclose(actual, original, rtol=2e-14, atol=2e-14)
    # Bound the FP32 material rounding against the independent FP64 result;
    # cancellation can make an individual off-diagonal entry almost zero.
    magnitude = np.einsum(
        "...ij,aj...->ai...", np.abs(rounded[ids]), np.abs(arrays.astype(np.float64))
    )
    bound = (
        8
        * np.finfo(np.float32).eps
        * np.einsum("aixyz,bixyz->ab", magnitude, np.abs(arrays.astype(np.float64)))
        / np.prod(shape)
    )
    assert np.all(np.abs(actual - expected) <= bound)
    for before, field in zip(arrays, fields):
        np.testing.assert_array_equal(cp.asnumpy(field.val), before)


@pytest.mark.gpu
def test_indexed_assembly_preserves_custom_operator_contract():
    from cufhom_core.general.cupy_cfield_assembly import indexed_assembly

    class CustomTensor(Tensor):
        def __call__(self, field):
            return 2 * field

    shape = (2, 3, 4)
    A = CustomTensor(
        name="custom",
        val=np.zeros((1,) + shape, np.int32),
        order=1,
        N=shape,
        Fourier=False,
        multype="sym21_indexed",
    )
    fields = [
        Tensor(
            name=str(k),
            val=np.ones((6,) + shape, np.float32),
            order=1,
            N=shape,
            Fourier=False,
        )
        for k in range(6)
    ]
    assert indexed_assembly(A, fields) is None


@pytest.mark.gpu
@pytest.mark.parametrize("bad_id", [-1, 1])
def test_indexed_assembly_rejects_out_of_range_material_ids(bad_id):
    if cp is None:
        pytest.skip("CuPy unavailable")
    shape = (3, 3, 3)
    A = Tensor(
        name="indexed",
        val=cp.zeros((1,) + shape, cp.int32),
        order=1,
        N=shape,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    A.material_table = cp.ones((1, 21), cp.float32)
    A.val.ravel()[7] = bad_id
    fields = [
        Tensor(
            name=str(k),
            val=cp.ones((6,) + shape, cp.float32),
            order=1,
            N=shape,
            Fourier=False,
            fft_form="r",
        )
        for k in range(6)
    ]
    with pytest.raises(ValueError, match="Nonfinite indexed assembly"):
        assembly_matrix_fused(A, fields)


@pytest.mark.gpu
def test_fused_generic_preserves_custom_field_multiplication():
    if cp is None:
        pytest.skip("CuPy unavailable")

    class CustomField(Tensor):
        def __mul__(self, other, *args, **kwargs):
            return 7.0

    shape = (3, 3, 3)
    fields = [
        CustomField(
            name=str(k),
            val=cp.ones((6,) + shape, cp.float32),
            order=1,
            N=shape,
            Fourier=False,
            fft_form="r",
        )
        for k in range(6)
    ]

    class Identity:
        N = np.array(shape)

        def __call__(self, field):
            return field

    actual = assembly_matrix_fused(Identity(), fields)
    np.testing.assert_array_equal(actual, np.full((6, 6), 7.0))
