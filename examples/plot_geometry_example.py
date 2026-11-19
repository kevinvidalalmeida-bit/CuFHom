"""Generate and render an illustrative fiber geometry for the README.

Run from the repository root after installing ``cufhom`` and its plotting extra::

    python -m pip install -e '.[plots]'
    python examples/plot_geometry_example.py

The geometry is generated specifically for this example, not from a campaign.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cufhom.geometry import GeometryRequest, generate_geometry  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402


def main():
    """Generate a reproducible geometry and render its smoothed voxel surface."""
    output = Path(__file__).resolve().parents[1] / "docs/assets/geometry-example.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cufhom-readme-") as folder:
        result_dir = Path(folder) / "geometry"
        result = generate_geometry(
            GeometryRequest(
                output=result_dir,
                box=12.0,
                ar=4.0,
                diameter=1.0,
                vf=0.05,
                a2=(0.6, 0.3),
                seed=572,
                threads=1,
                grid=72,
                shape="cylinder",
                solver="a2",
                a2_tolerance=0.04,
                voxel_a2_tolerance=0.08,
                vf_tolerance=0.01,
                overlap_tolerance=0.002,
                verbose=False,
            )
        )
        if not result["accepted"]:
            raise RuntimeError("Example geometry did not pass its checks")
        with np.load(result_dir / "geometry.npz", allow_pickle=False) as data:
            phase = data["phase"].astype(float)

    # Smoothing is only for the displayed surface; the underlying field is binary.
    field = np.pad(gaussian_filter(phase, 0.5), 1)
    vertices, faces, _, _ = marching_cubes(field, level=0.5, step_size=2)
    vertices = (vertices - 1.0) / phase.shape[0]
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.5, -0.6, 0.8])
    light /= np.linalg.norm(light)
    intensity = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    base = np.array(to_rgb("#1687ac"))
    colors = base[None, :] * intensity[:, None] + np.array([0.04, 0.08, 0.10])
    colors = np.clip(colors, 0, 1)

    fig = plt.figure(figsize=(12, 5.1), facecolor="#f7f9fc")
    ax = fig.add_axes((0.015, 0.03, 0.55, 0.93), projection="3d")
    ax.set_facecolor("#f7f9fc")
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0))
    for i in (0, 1):
        for j in (0, 1):
            for axis in range(3):
                points = np.array([[i, j, 0], [i, j, 1]], dtype=float)
                points = np.roll(points, axis, axis=1)
                ax.plot(*points.T, color="#aab9c8", linewidth=0.7, alpha=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

    fig.text(0.56, 0.77, "CuFHom", color="#10283f", fontsize=35, fontweight="bold")
    fig.text(
        0.56,
        0.64,
        "Computational homogenization\nof voxelized materials",
        color="#36536a",
        fontsize=16,
        linespacing=1.45,
    )
    fig.add_artist(plt.Line2D([0.56, 0.95], [0.54, 0.54], color="#c7d4df", linewidth=1))
    fig.text(
        0.56,
        0.42,
        "01   Periodic microstructures",
        color="#213e56",
        fontsize=12,
    )
    fig.text(
        0.56, 0.33, "02   Linear and nonlinear solvers", color="#213e56", fontsize=12
    )
    fig.text(0.56, 0.24, "03   CPU and GPU execution", color="#213e56", fontsize=12)
    fig.text(
        0.56, 0.11, "Illustrative geometry · 72³ voxels", color="#61788a", fontsize=10
    )
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
