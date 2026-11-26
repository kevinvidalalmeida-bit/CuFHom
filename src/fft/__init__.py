"""Linear FFT homogenization of periodic elastic materials."""

from .grids import select_grid
from .krylov import ConvergenceError
from .linear.common import IndexedMaterial, SolveOptions
from .linear.framework import homogenize
from .linear.solver import solve_homogenization

__all__ = [
    "select_grid",
    "ConvergenceError",
    "IndexedMaterial",
    "SolveOptions",
    "homogenize",
    "solve_homogenization",
]
