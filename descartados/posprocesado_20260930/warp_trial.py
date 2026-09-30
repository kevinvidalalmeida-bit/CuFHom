from contextlib import contextmanager
import importlib
import numpy as np
import cupy as cp
from profile_g09_linear import parser, run
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction
postprocess = importlib.import_module('cufhom_core.postprocess')
reduction = cp.ReductionKernel('T x, T y', 'float64 z', '(double)x * (double)y',
    'a + b', 'z = a', '0', 'cufhom_real_energy_dot')

def fused_assembly(Afun, solutions, batch_size=1, method='scalar'):
    sol = solutions
    stress = [Afun(s) for s in sol]
    reducer = CupyWarpCGReduction(sol[0], max_blocks=16384)
    dots = []
    for a in stress:
        for s in sol:
            reducer.dot_kernel((reducer.blocks,), (reducer.threads,),
                (a.val, s.val, reducer.partial, np.int64(s.val.size)))
            dots.append(cp.sum(reducer.partial, dtype=cp.float64))
    return cp.asnumpy(cp.stack(dots)).reshape(len(sol),len(sol))/float(np.prod(Afun.N))

@contextmanager
def candidate():
    original = postprocess.assembly_matrix
    postprocess.assembly_matrix = fused_assembly
    try: yield
    finally: postprocess.assembly_matrix = original

from contextlib import nullcontext
run(parser().parse_args(), variants={'current': ({},nullcontext), 'warp_energy': ({}, candidate)})
