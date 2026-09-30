from contextlib import contextmanager
import importlib
import numpy as np
import cupy as cp
from profile_g09_linear import parser, run
postprocess = importlib.import_module('cufhom_core.postprocess')
reduction = cp.ReductionKernel('T x, T y', 'float64 z', '(double)x * (double)y',
    'a + b', 'z = a', '0', 'cufhom_real_energy_dot')

def fused_assembly(Afun, solutions, batch_size=1, method='scalar'):
    sol = solutions
    stress = [Afun(s) for s in sol]
    dots = [reduction(a.val, s.val) for a in stress for s in sol]
    return cp.asnumpy(cp.stack(dots)).reshape(len(sol),len(sol))/float(np.prod(Afun.N))

@contextmanager
def candidate():
    original = postprocess.assembly_matrix
    postprocess.assembly_matrix = fused_assembly
    try: yield
    finally: postprocess.assembly_matrix = original

from contextlib import nullcontext
run(parser().parse_args(), variants={'current': ({},nullcontext), 'fused_energy': ({}, candidate)})
