"""Inspect every slot loaded by the production HEX8 CUDA plane loader.

CPU tests require no CUDA imports. GPU tests intentionally inspect padding
lanes too: checking only valid nodal outputs misses the original bad reads.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

_TY, _TZ = 8, 32
_HP = (_TY + 2) * (_TZ + 2)
_SHAPES = [
    (2, 2, 2),
    (4, 4, 4),
    (3, 4, 32),
    (3, 8, 16),
    (3, 5, 17),
    (3, 7, 31),
    (3, 8, 32),
    (5, 9, 35),
]


def _expected_halos(shape):
    """Independent NumPy periodic indexing, including all padding/halo slots."""
    nx, ny, nz = shape
    planes = []
    for xp in range(-1, nx + 1):
        rows = []
        for y0 in range(0, ny, _TY):
            tiles = []
            for z0 in range(0, nz, _TZ):
                y = np.arange(y0 - 1, y0 + _TY + 1) % ny
                z = np.arange(z0 - 1, z0 + _TZ + 1) % nz
                tiles.append(((xp % nx * ny + y[:, None]) * nz + z).ravel())
            rows.append(tiles)
        planes.append(rows)
    return np.asarray(planes)


def test_cpu_original_single_wrap_escapes_allocation():
    """Reachable slot: thread 83, second loop trip, first x=-1 plane of 4^3."""

    def original_wrap(q, size):
        q += size if q < 0 else 0
        return q - size if q >= size else q

    k, nx, ny, nz = 339, 4, 4, 4
    raw_y, raw_z = k // (_TZ + 2) - 1, k % (_TZ + 2) - 1
    y, z = original_wrap(raw_y, ny), original_wrap(raw_z, nz)
    address = ((nx - 1) * ny + y) * nz + z
    assert (raw_y, raw_z, y, z, address) == (8, 32, 4, 28, 92)
    assert address >= nx * ny * nz
    assert _expected_halos((nx, ny, nz))[0, 0, 0, k] == 48


@pytest.fixture(scope="module")
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp


def _production_loader():
    """Read the actual CUDA helper without importing or constructing a solver."""
    path = Path(__file__).resolve().parents[3] / "src/fft/linear/hex8_general.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_CUDA_INTERIOR"
            for target in node.targets
        ):
            source = ast.literal_eval(node.value)
            return source.split('extern "C" __global__', 1)[0]
    raise AssertionError("Production HEX8 interior loader not found")


_PROBE = r"""
extern "C" __global__ void inspect_hex8_plane(const int* index,const XT* u,
 int* labels,REAL* values,int nx,int ny,int nz){
 __shared__ REAL U[3][3][HP];
 __shared__ int L[HP];
 const int plane=blockIdx.z/3,mode=blockIdx.z%3,s=plane%3;
 const long long n=(long long)nx*ny*nz;
 // Exercise both buffers together and each of the two nullable arguments.
 hex8_plane(index,u,mode==2?nullptr:U,mode==1?nullptr:L,s,plane-1,
            blockIdx.y*TY,blockIdx.x*TZ,nx,ny,nz,n);
 __syncthreads();
 const long long b=(((long long)plane*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*3+mode;
 for(int k=threadIdx.y*TZ+threadIdx.x;k<HP;k+=TZ*TY){
  if(mode!=1)labels[b*HP+k]=L[k];
  if(mode!=2)for(int c=0;c<3;c++)values[(b*3+c)*HP+k]=U[s][c][k];
 }
}
"""


@pytest.mark.gpu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_production_loader_matches_modulo_for_every_halo_slot(cuda, shape, dtype):
    cp = cuda
    nx, ny, nz = shape
    n = nx * ny * nz
    real = "float" if dtype == "float32" else "double"
    source = (
        f"#define REAL {real}\n#define XT {real}\n"
        "#define TOREAL(x) ((REAL)(x))\n"
        f"#define IS_DOUBLE {int(dtype == 'float64')}\n" + _production_loader() + _PROBE
    )
    kernel = cp.RawKernel(source, "inspect_hex8_plane", options=("--std=c++11",))

    # Unique values expose wrong coordinates and component crossovers. A guard
    # tail makes the old invalid addresses readable sentinels, so a regression
    # fails equality instead of poisoning the CUDA context for subsequent tests.
    guard = _TY * nz + _TZ
    labels = np.full(n + guard, -1, np.int32)
    labels[:n] = np.arange(n, dtype=np.int32)
    values = np.full(3 * n + guard, -1, dtype=dtype)
    values[: 3 * n] = np.arange(1, 3 * n + 1, dtype=dtype)
    expected = _expected_halos(shape)
    assert expected.shape[-1] == _HP
    got_labels = cp.full((*expected.shape[:-1], 3, _HP), -1, cp.int32)
    got_values = cp.full((*expected.shape[:-1], 3, 3, _HP), np.nan, dtype)
    kernel(
        ((nz + _TZ - 1) // _TZ, (ny + _TY - 1) // _TY, (nx + 2) * 3),
        (_TZ, _TY, 1),
        (
            cp.asarray(labels),
            cp.asarray(values),
            got_labels,
            got_values,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        ),
    )
    got_labels, got_values = cp.asnumpy(got_labels), cp.asnumpy(got_values)
    expected_values = np.stack(
        [expected + 1 + c * n for c in range(3)], axis=-2
    ).astype(dtype)
    for mode in (0, 2):
        np.testing.assert_array_equal(got_labels[..., mode, :], expected)
    for mode in (0, 1):
        np.testing.assert_array_equal(got_values[..., mode, :, :], expected_values)
    assert np.all(got_labels[..., 1, :] == -1)
    assert np.isnan(got_values[..., 2, :, :]).all()
