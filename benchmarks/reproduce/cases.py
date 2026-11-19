"""Prepare the seven fixed paper microstructures without historical campaigns.

Material matrices use Mandel order (11, 22, 33, 23, 13, 12), packed as the
21 upper-triangular entries. All lengths refer to a periodic unit cube.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

INPUTS = Path(__file__).resolve().parent / "inputs"
CASES = (
    "particles",
    "polycrystal",
    "gyroid",
    "random_short_fibres",
    "laminate",
    "hashin",
    "public",
)


def array_sha256(array):
    """Hash contiguous array bytes, independently of container compression."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_json(name):
    return json.loads((INPUTS / name).read_text(encoding="utf-8"))


def _sample_family(case, n, definition):
    """Evaluate the frozen continuous geometry at voxel centres in XYZ order."""
    x = (np.arange(n) + 0.5) / n
    index = np.zeros((n, n, n), np.int32)
    tree = (
        cKDTree(np.asarray(definition["polycrystal"]["seeds"]), boxsize=1.0)
        if case == "polycrystal"
        else None
    )
    for start in range(0, n, 8):
        X, Y, Z = np.meshgrid(x[start : start + 8], x, x, indexing="ij")
        if case == "particles":
            mask = np.zeros(X.shape, bool)
            for center, radius in zip(
                definition[case]["centers"], definition[case]["radii"], strict=True
            ):
                distances = [
                    np.abs((v - a + 0.5) % 1 - 0.5)
                    for v, a in zip((X, Y, Z), center, strict=True)
                ]
                mask |= sum(v * v for v in distances) < radius * radius
            index[start : start + 8] = mask
        elif case == "polycrystal":
            points = np.column_stack([v.ravel() % 1 for v in (X, Y, Z)])
            index[start : start + 8] = tree.query(points, workers=1)[1].reshape(X.shape)
        elif case == "gyroid":
            a, b, c = [2 * np.pi * v for v in (X, Y, Z)]
            value = (
                np.sin(a) * np.cos(b) + np.sin(b) * np.cos(c) + np.sin(c) * np.cos(a)
            )
            index[start : start + 8] = value > definition[case]["threshold"]
        else:
            normal = definition[case]["normal_integer"]
            value = normal[0] * X + normal[1] * Y + normal[2] * Z
            index[start : start + 8] = (
                value + definition[case]["offset"]
            ) % 1 < definition[case]["fraction"]
    return index


def _sample_hashin(n):
    """Evaluate the neutral three-phase coated sphere at voxel centres."""
    x = (np.arange(n) + 0.5) / n - 0.5
    radius2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    index = np.zeros((n, n, n), np.int32)
    index[radius2 < 0.4**2] = 1
    index[radius2 < (0.4 * 0.14 ** (1 / 3)) ** 2] = 2
    return index


def prepare_case(case, n, dtype="float64"):
    """Return ``(index, packed_stiffness, metadata)`` for a fixed paper case.

    Parameters
    ----------
    case : str
        One of :data:`CASES`. The six comparison cells are distinct from the
        public image; aligned-fibre experiments are not substituted for them.
    n : int
        Cubic grid edge, at least two. Continuous cells are sampled afresh;
        the public image uses nearest voxel-centre resampling from 192 cubed.
    dtype : str or numpy.dtype
        Float64 or float32 material storage. Geometry indices remain int32.
        Float32 is an explicit cast of the archived nominal float64 table.

    Notes
    -----
    Copper orientations and anisotropic matrices are frozen in the input
    definitions and material table. Metadata reference values describe the
    historical isotropic preconditioner input, not a tensor-error reference.
    This function does not initialize CUDA, cache arrays, or write results.
    """
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Grid edge n must be an integer >= 2")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Material dtype must be float32 or float64")
    material = _read_json("materials.json")[case]
    table = np.ascontiguousarray(material["table"], dtype=dtype)
    sampling = "fixed continuous definitions evaluated at voxel centres"
    if case == "public":
        with np.load(INPUTS / "public_phase192.npz", allow_pickle=False) as data:
            raw = data["phase"]
        source = _read_json("public_source.json")
        if array_sha256(raw) != source["raw_phase_sha256"]:
            raise ValueError("Public phase image failed its source checksum")
        axes = np.floor((np.arange(n) + 0.5) * 192 / n).astype(int)
        index = np.ascontiguousarray(
            raw.transpose(2, 1, 0)[np.ix_(axes, axes, axes)], dtype=np.int32
        )
        sampling = "XYZ transpose of source image; floor((i+0.5)*192/n) per axis"
    elif case == "hashin":
        index = _sample_hashin(n)
    elif case == "random_short_fibres":
        from cufhom.geometry.voxel import rasterize_cylinders

        definition = _read_json("random_short_fibres.json")
        index, _, _, _ = rasterize_cylinders(
            np.asarray(definition["centers"]) * n - 0.5,
            np.asarray(definition["directions"]),
            definition["length"] * n,
            definition["diameter"] * n / 2,
            n,
        )
        index = index.astype(np.int32)
    else:
        index = _sample_family(case, n, _read_json("periodic_families.json"))
    metadata = {
        "case": case,
        "n": int(n),
        "dtype": str(dtype),
        "reference": material["reference"],
        "sampling": sampling,
        "mandel_order": ["11", "22", "33", "23", "13", "12"],
        "index_sha256": array_sha256(index),
        "table_sha256": array_sha256(table),
        "nominal_table_sha256": material["table_sha256"],
    }
    return index, table, metadata


def load_case(case, n, dtype="float64"):
    """Alias for :func:`prepare_case` used by the campaign driver."""
    return prepare_case(case, n, dtype=dtype)
