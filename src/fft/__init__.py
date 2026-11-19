"""Linear elasticity and incremental J2 FFT homogenization."""

from .krylov import NonlinearConvergenceError
from .linear.common import IndexedMaterial, SolveOptions
from .linear.framework import homogenize
from .linear.solver import solve_homogenization
from .nonlinear.solver import solve_nonlinear_homogenization

__all__ = [
    "solve_homogenization",
    "solve_nonlinear_homogenization",
    "NonlinearConvergenceError",
    "homogenize",
    "IndexedMaterial",
    "SolveOptions",
]
