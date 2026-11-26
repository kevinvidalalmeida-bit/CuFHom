"""Check linear homogenization with public precision choices and default controls."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from cufhom.fft import homogenize
from cufhom.geometry import GeometryRequest, generate_geometry


def main():
    """Check an analytic homogeneous solid and optionally generated fiber cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Also generate straight and flexible cells with two isotropic phases",
    )
    args = parser.parse_args()
    E, nu = 3.0, 0.3
    mu = E / (2 * (1 + nu))
    lame = E * nu / ((1 + nu) * (1 - 2 * nu))
    expected = 2 * mu * np.eye(6)
    expected[:3, :3] += lame
    result = homogenize(
        np.zeros((4, 4, 4), dtype=np.int32),
        expected[None],
        precision="fp64",
        backend="numpy",
    )
    np.testing.assert_allclose(result["C"], expected, atol=1e-10, rtol=1e-10)
    assert all(audit["accepted"] for audit in result["audits"])
    print("PASS: homogeneous linear stiffness, FP64, default controls, 4^3 CPU")

    if args.geometry:
        with TemporaryDirectory(prefix="cufhom-smoke-") as folder:
            for model in ("straight", "flexible"):
                output = Path(folder) / model
                summary = generate_geometry(
                    GeometryRequest(
                        output=output,
                        fiber_model=model,
                        shape="cylinder" if model == "straight" else "capsule",
                        box=12.0,
                        ar=2.0,
                        vf=0.02,
                        grid=36,
                        segments=2,
                        a2=(1.0, 0.0),
                        seed=572,
                        verbose=False,
                    )
                )
                assert summary["accepted"], summary
                with np.load(output / "geometry.npz", allow_pickle=False) as data:
                    phase = data["phase"]
                # Isotropic fibers: orientation does not change these two tensors.
                result = homogenize(
                    phase,
                    np.stack([expected, 20 * expected]),
                    precision="fp64",
                    backend="numpy",
                )
                assert all(audit["accepted"] for audit in result["audits"])
                assert np.linalg.eigvalsh(result["C"]).min() > 0
                print(
                    f"PASS: {model}, {summary['accepted_count']} fibers, "
                    f"Vf={summary['vf']:.6f}, linear FP64 CPU"
                )


if __name__ == "__main__":
    main()
