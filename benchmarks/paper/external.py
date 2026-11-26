"""Independent six-load reference adapters with an explicit audited-call clock.

FANS here is the PyTorch FANS preconditioner distributed in UNOCG, as in the
paper's GPU comparisons. It is distinct from the standalone C++ FANS program.
External packages are optional and are never replaced by CuFHom implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .cases import CASES, prepare_case
from .run import environment, plain, strict_record

PERM = np.array([0, 1, 2, 5, 4, 3])


def unpack(table):
    """Expand packed symmetric matrices in CuFHom Mandel order."""
    matrices = np.zeros((len(table), 6, 6), dtype=np.float64)
    i, j = np.triu_indices(6)
    matrices[:, i, j] = table
    matrices[:, j, i] = table
    return matrices


def _source_path(path, package):
    if path is not None:
        path = Path(path).resolve()
        if not (path / package / "__init__.py").is_file():
            raise ValueError(f"Expected {package}/__init__.py beneath {path}")
        sys.path.insert(0, str(path))


def _external_identity(package):
    base = Path(package.__file__).resolve().parent
    commit = None
    if (base.parent / ".git").exists():
        revision = subprocess.run(
            ["git", "-C", str(base.parent), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    return {
        "source": str(base),
        "git_commit": commit,
        "python_sources_sha256": {
            str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(base.rglob("*.py"))
        },
    }


def _general_material(torch, config):
    from unocg.materials import Material

    class GeneralLinearElasticity(Material):
        """Apply the fixed grain tensors at the official material interface."""

        def get_stress(self, strain, param_fields, *args, **kwargs):
            """Return stress and no constitutive history for linear elasticity."""
            matrix = param_fields.reshape(
                param_fields.shape[0], 6, 6, *param_fields.shape[-3:]
            )
            return torch.einsum("bijzyx,blgjzyx->blgizyx", matrix, strain), None

        def get_tangent_vp(self, strain0, strain, param_fields, *args, **kwargs):
            """Apply the unchanged tangent to a strain increment."""
            return self.get_stress(strain, param_fields, *args, **kwargs)[0]

        def get_tangent(self, strain, param_fields, *args, **kwargs):
            """Expose the material tangent at every quadrature point."""
            matrix = param_fields.reshape(
                param_fields.shape[0], 6, 6, *param_fields.shape[-3:]
            )
            return matrix[:, None, None].expand(
                matrix.shape[0],
                strain.shape[1],
                strain.shape[2],
                6,
                6,
                *matrix.shape[-3:],
            )

    return GeneralLinearElasticity(n_dim=3, **config)


def fans(index, table, eta, *, source=None, maxiter=3000):
    """Run FANS FP64 with fresh nodal Linf audits and corrected assembly.

    Inputs are host arrays. Imports and CUDA-context initialization precede the
    clock; material preparation, six zero-start solves, fresh residuals, host
    transfers and assembly are timed together. No trained UNO weights are used.
    """
    if any(n % 2 for n in index.shape):
        raise ValueError("The pinned FANS reference requires even cubic grids")
    if not 0 < eta < 1 or maxiter < 1:
        raise ValueError("eta must be in (0,1) and maxiter must be positive")
    _source_path(source, "unocg")
    import torch
    import unocg
    from unocg.materials.mechanical import LinearElasticity
    from unocg.preconditioners.torch import FansPreconditioner
    from unocg.problems import BC
    from unocg.problems.mechanical import MechanicalProblem
    from unocg.solvers.torch import CgSolver

    if not torch.cuda.is_available():
        raise RuntimeError("FANS reference requires a CUDA-enabled PyTorch environment")
    torch.zeros(1, device="cuda").sum()
    torch.cuda.synchronize()
    started = time.perf_counter()
    matrices = unpack(table)
    lame, shear = matrices[:, 0, 1], matrices[:, 3, 3] / 2
    isotropic = 2 * shear[:, None, None] * np.eye(6)
    isotropic[:, :3, :3] += lame[:, None, None]
    anisotropic = not np.allclose(matrices, isotropic, rtol=1e-14, atol=1e-13)
    config = {"device": "cuda:0", "dtype": torch.float64}
    material = (
        _general_material(torch, config)
        if anisotropic
        else LinearElasticity(n_dim=3, **config)
    )
    image = np.ascontiguousarray(index.transpose(2, 1, 0))
    problem = MechanicalProblem(
        image.shape, material=material, bc=BC.PERIODIC, quad_degree=2
    )
    if anisotropic:
        external = matrices[:, PERM][:, :, PERM]
        params = torch.tensor(external.reshape(-1, 36), **config)
        labels = torch.tensor(image, device="cuda:0", dtype=torch.long)
        fields = params[labels].movedim(-1, 0).unsqueeze(0).contiguous()
        reference = torch.tensor(external.mean(0), **config)
    else:
        params = torch.tensor(np.column_stack([lame, shear]), **config)
        labels = torch.tensor(image, device="cuda:0").unsqueeze(0)
        fields = problem.get_param_fields(labels, params)
        reference = problem.stiffness(params.mean(-2))
    module = CgSolver(problem, FansPreconditioner(problem, reference)).get_module(
        **config, rtol=eta, atol=0.0, err_eps=0.0
    )
    residual = problem.get_residual_module(**config)
    field_module = problem.get_field_module(**config)
    count = [0]

    def update(*unused):
        count[0] += 1
        if count[0] > maxiter:
            raise RuntimeError("FANS exceeded its per-load iteration budget")

    module.iteration_layers[0].register_forward_hook(update)
    solutions, residuals, loads = [], [], []
    with torch.inference_mode():
        for i in range(6):
            count[0] = 0
            macro = torch.tensor(np.eye(6)[i : i + 1, PERM], **config)
            guess = module.zero_guess(fields, macro)
            rhs = float(residual(guess, fields, macro).abs().max())
            solution = module(guess, fields, macro) if rhs else guess
            solution -= solution.mean((-3, -2, -1), keepdim=True)
            fresh = residual(solution, fields, macro)
            absolute = float(fresh.abs().max())
            relative = absolute / rhs if rhs else absolute
            stress = field_module(solution, fields, macro)
            loads.append(
                {
                    "load": i,
                    "updates": count[0],
                    "rhs_norm": rhs,
                    "fresh_relative": relative,
                    "accepted": bool(np.isfinite(relative) and relative <= eta),
                    "stress": stress[0, 0, 3:9].mean((-3, -2, -1))[PERM].cpu().numpy(),
                }
            )
            solutions.append(solution.cpu().numpy().ravel())
            residuals.append(fresh.cpu().numpy().ravel())
    direct = np.asarray([load["stress"] for load in loads]).T
    correction = np.asarray([[np.dot(u, r) for r in residuals] for u in solutions])
    corrected = direct - correction
    torch.cuda.synchronize()
    return {
        "audited_call_s": time.perf_counter() - started,
        "status": "accepted"
        if all(row["accepted"] for row in loads)
        else "residual_not_met",
        "C": corrected,
        "C_direct": direct,
        "loads": loads,
        "norm": "volume-normalized HEX8 nodal Linf",
        "iterations": [row["updates"] for row in loads],
        "external_source": _external_identity(unocg),
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }


def ffthompy(index, table, eta, *, source=None, precision="fp64", maxiter=3000):
    """Run the upstream GaNi CG driver and fresh projected residuals on an odd grid.

    The clock includes material construction, projector, six solves, assembly
    and a fresh IEEE FP64 audit of the actual stored coefficient field. FP32
    labels coefficient/projector storage, not a guarantee about every FFT
    intermediate. Historical FFTHomPy timing excluded the last audit.
    """
    if any(n % 2 == 0 for n in index.shape):
        raise ValueError("Use odd grids for matched FFTHomPy/GaNi comparisons")
    if not 0 < eta < 1 or maxiter < 1:
        raise ValueError("eta must be in (0,1) and maxiter must be positive")
    if precision not in ("fp64", "fp32"):
        raise ValueError("FFTHomPy reference precision must be fp64 or fp32")
    _source_path(source, "ffthompy")
    # Restore aliases used by the pinned upstream without editing its sources.
    for name, value in (("float", float), ("int", int), ("complex", complex)):
        np.__dict__.setdefault(name, value)
    if not hasattr(time, "clock"):
        time.clock = time.process_time
    import scipy

    if not hasattr(scipy, "setdiff1d"):
        scipy.setdiff1d = np.setdiff1d
    package = importlib.import_module("ffthompy")
    from ffthompy.general.solver import linear_solver
    from ffthompy.postprocess import add_macro2minimizer, assembly_matrix
    from ffthompy.projections import elasticity
    from ffthompy.tensors import DFT, Operator, Tensor

    started = time.perf_counter()
    dtype = np.float64 if precision == "fp64" else np.float32
    field = np.ascontiguousarray(
        np.moveaxis(unpack(table)[index], (-2, -1), (0, 1)), dtype=dtype
    )
    grid, lengths = tuple(index.shape), np.ones(3)
    _, hydro, shear, _, _ = elasticity(grid, lengths, NyqNul=True)
    projector = hydro + shear
    if precision == "fp32":
        projector.val = projector.val.astype(np.float32)
    forward = DFT(name="FN", inverse=False, N=grid)
    inverse = DFT(name="FiN", inverse=True, N=grid)
    projection = Operator(name="G1", mat=[[inverse, projector, forward]])
    material = Tensor(
        name="A", val=field, order=2, N=grid, Y=lengths, multype=21, Fourier=False
    )
    action = Operator(name="GA", mat=[[projection, material]])
    solutions, fluctuations, counts = [], [], []
    for i in range(6):
        macro = np.eye(6)[i]
        mean = Tensor(name="EN", N=grid, shape=(6,), Fourier=False)
        mean.set_mean(macro)
        if precision == "fp32":
            mean.val = mean.val.astype(np.float32)
        rhs = action(-mean)
        norm = float(rhs * rhs) ** 0.5
        if norm:
            fluctuation, info = linear_solver(
                solver="CG",
                Afun=action,
                B=rhs,
                x0=mean.zeros_like(name="x0"),
                par={"tol": eta * norm, "maxiter": maxiter},
            )
        else:
            fluctuation, info = mean.zeros_like(name="x0"), {"kit": 0}
        if precision == "fp64":
            total = add_macro2minimizer(fluctuation, macro)
        else:
            if not np.allclose(fluctuation.mean(), 0, rtol=0, atol=1e-5):
                raise ValueError("FFTHomPy FP32 fluctuation is not zero mean")
            total = fluctuation + mean
        solutions.append(total)
        fluctuations.append((fluctuation, mean, norm))
        counts.append(int(info["kit"]))
    tensor = np.asarray(assembly_matrix(material, solutions), dtype=np.float64)
    projection64 = Operator(name="G64", mat=[[inverse, hydro + shear, forward]])
    material64 = Tensor(
        name="A64",
        val=field.astype(np.float64),
        order=2,
        N=grid,
        Y=lengths,
        multype=21,
        Fourier=False,
    )
    action64 = Operator(name="GA64", mat=[[projection64, material64]])
    loads = []
    for i, (fluctuation, mean, norm) in enumerate(fluctuations):
        rhs64 = action64(-mean)
        denominator = float(rhs64 * rhs64) ** 0.5
        fresh = rhs64 - action64(fluctuation)
        absolute = float(fresh * fresh) ** 0.5
        relative = absolute / denominator if denominator else absolute
        loads.append(
            {
                "load": i,
                "updates": counts[i],
                "rhs_norm": denominator,
                "fresh_relative": relative,
                "accepted": bool(np.isfinite(relative) and relative <= eta),
            }
        )
    return {
        "audited_call_s": time.perf_counter() - started,
        "status": "accepted"
        if all(row["accepted"] for row in loads)
        else "residual_not_met",
        "C": tensor,
        "loads": loads,
        "iterations": counts,
        "norm": "FFTHomPy projected L2 with its own normalization",
        "external_source": _external_identity(package),
    }


def main(argv=None):
    """Run one reference in a separate process and write a new JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("fans", "ffthompy"))
    parser.add_argument("--case", choices=CASES, default="public")
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--source", type=Path, help="Optional upstream checkout root")
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.eta < 1 or args.warm < 0:
        parser.error("eta must be in (0,1) and warm must be nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.solver == "fans" and args.precision != "fp64":
        parser.error("FANS FP32 is excluded: its physical tolerance was not met")
    index, table, metadata = prepare_case(args.case, args.grid)
    result = {
        "solver": args.solver,
        "precision": args.precision,
        "eta": args.eta,
        "input": metadata,
        "environment": environment(),
        "timing_scope": "host inputs already loaded; preparation through audited tensor",
        "calls": [],
    }
    try:
        for i in range(1 + args.warm):
            if args.solver == "fans":
                call = fans(index, table, args.eta, source=args.source)
            else:
                call = ffthompy(
                    index, table, args.eta, source=args.source, precision=args.precision
                )
            call["state"] = "first" if i == 0 else "warm"
            result["calls"].append(call)
        result["status"] = (
            "accepted"
            if all(c["status"] == "accepted" for c in result["calls"])
            else "residual_not_met"
        )
    except Exception as error:
        result.update(status="failed", error=repr(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            strict_record(result), stream, indent=2, default=plain, allow_nan=False
        )
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
