"""Effective stiffness from the solved strain fields (GaNi, primal)."""
import numpy as np
from cufhom_core.general.base import Timer
from cufhom_core.general.cupy_cfield_assembly import indexed_assembly, indexed_assembly_streamed
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import get_array_module, to_backend_array, to_host_array


def postprocess(pb, A, solutions, results):
    """Assemble AH_ij = <A e_i, e_j> and store it with the load results."""
    tim = Timer(name='postprocessing')
    print('\npostprocessing')
    name = 'AH_GaNi_primal'
    print(('calculating: ' + name))
    if str(pb.solve.get('postprocess_assembly', 'gemm')) == 'fused':
        AH = assembly_matrix_fused(A, solutions)
    else:
        AH = assembly_matrix_gemm(A, solutions, batch_size=int(pb.solve.get('postprocess_batch_size', 6)))
    tim.measure()
    output = {'res_primal': results, 'mat_primal': {name: AH}}
    if pb.solve.get('store_solution_fields', True):
        output['sol_primal'] = solutions
    pb.output.update(output)


def assembly_matrix_fused(Afun, sol):
    """GPU assembly with FP64 products and sums, one field at a time.

    The indexed material is contracted without stress buffers; otherwise each
    stress is reduced against every field with the warp reduction. Every
    entry is evaluated: no symmetry or material assumption is used.
    """
    dim = len(sol)
    matrix = indexed_assembly(Afun, sol)
    if matrix is None and any(isinstance(field.val, np.ndarray) for field in sol):
        matrix = indexed_assembly_streamed(Afun, sol)
    if matrix is not None:
        matrix = to_host_array(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite indexed assembly: check material IDs and fields')
        return matrix/float(np.prod(sol[0].N))

    A_sol = [Afun(sol[ii]) for ii in range(dim)]
    if all(
        type(field) is Tensor and CupyWarpCGReduction.supports(field, {})
        and field.val.dtype == sol[0].val.dtype
        and field.val.shape == sol[0].val.shape
        and field.fft_form == sol[0].fft_form
        and np.array_equal(field.N, sol[0].N)
        for field in A_sol + list(sol)
    ):
        xp = get_array_module(sol[0].val)
        reduction = CupyWarpCGReduction(sol[0])
        entries = []
        for stress in A_sol:
            for field in sol:
                reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                    (stress.val, field.val, reduction.partial, np.int64(reduction.total)))
                entries.append(xp.sum(reduction.partial, dtype=xp.float64))
        return to_host_array(xp.stack(entries)).reshape(dim, dim)/float(np.prod(sol[0].N))

    AH = np.zeros([dim, dim])
    for ii in range(dim):
        for jj in range(dim):
            AH[ii, jj] = A_sol[ii] * sol[jj]
    return AH


def assembly_matrix_gemm(Afun, sol, batch_size=6):
    """Batched material action and one matrix product per batch (CPU)."""
    dim = len(sol)
    xp = get_array_module(*[item.val for item in sol])
    sol_all_val = xp.stack([item.val for item in sol], axis=1)
    sol_all = sol[0].copy(name='sol_batch', val=sol_all_val, order=2)

    spatial_axes = tuple(range(2, sol_all.val.ndim))
    sol_mat = xp.transpose(sol_all.val, (1, 0) + spatial_axes).reshape((dim, -1))

    AH = np.zeros([dim, dim])
    prodN = float(np.prod(Afun.N))
    batch_size = max(1, min(int(batch_size), dim))
    for start in range(0, dim, batch_size):
        stop = min(start + batch_size, dim)
        sol_chunk = sol_all.copy(
            name='sol_batch_{0}_{1}'.format(start, stop),
            val=sol_all.val[:, start:stop, ...],
            order=2,
        )
        A_chunk = Afun(sol_chunk)
        xp_block = get_array_module(A_chunk.val, sol_all.val)
        A_mat = xp_block.transpose(A_chunk.val, (1, 0) + spatial_axes).reshape((stop - start, -1))
        sol_rhs = sol_mat
        if xp_block is not xp:
            sol_rhs = to_backend_array(sol_mat, prefer_backend='cupy')
        block = (A_mat @ sol_rhs.T).real / prodN
        AH[start:stop, :] = to_host_array(block)
    return AH
