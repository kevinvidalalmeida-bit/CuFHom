"""Process-local experimental controls; numerical production sources stay unchanged.

Extracted from the original paired study, with campaign wrappers removed.
Controls deliberately use private solver interfaces and fail if those change.
Run each control in its own process, never in a production application.
"""

import numpy as np


def patch_gani_direct_loads(counts):
    """Build all six load-stress components instead of reusing phase spectra.

    Returning None selects the existing six-component cuFFT load path. The
    operator, reference, Krylov arithmetic and fresh acceptance are unchanged.
    The caller also checks each load's native phase_spectra_rhs diagnostic.
    """
    from cufhom.fft.linear import coordinates

    def direct_load(*args):
        counts["direct_load_evaluations"] += 1
        return None

    coordinates._phase_spectra = direct_load


def patch_hex8(variant, counts):
    """Install one HEX8 gather, vector-traversal or inverse-symbol control."""
    import cupy as cp
    import numpy as np
    from cufhom.fft.linear import hex8_fused as f
    from cufhom.fft.linear import hex8_general as h

    original_init = h.Hex8Operator.__init__

    def initialize(op, *args, **kwargs):
        original_init(op, *args, **kwargs)
        if op.xp is cp:
            counts["operators"] += 1
            counts["n"] = int(op.n)
            counts["interface_nodes"] = int(op.interface_nodes.size)
            if variant == "inverse_rebuild":
                counts["tables_released"] += int(op.release_symbol_table())

    h.Hex8Operator.__init__ = initialize

    if variant == "gather":

        def launch(op, mode, real, u, out, partial, state, macro=None):
            if not hasattr(op, "_experimental_all_nodes"):
                op._experimental_all_nodes = cp.arange(op.n, dtype=cp.int64)
            ke = (
                op.ke_df
                if isinstance(real, str)
                else op.ke64
                if real == np.float64
                else op.ke
            )
            blocks = (op.n + 127) // 128
            counts["gather_launches"] += 1
            h._action_kernel("interface", mode, real, u.dtype, out.dtype)(
                (blocks,),
                (128,),
                (
                    op.index,
                    ke,
                    op._experimental_all_nodes,
                    u,
                    out,
                    partial,
                    state,
                    op.force64,
                    op._partial if macro is None else macro,
                    np.int64(op.n),
                    np.int64(op.n),
                    np.int32(op.shape[1]),
                    np.int32(op.shape[2]),
                    np.int32(op.shape[0]),
                    np.int32(op.phases),
                ),
            )
            return blocks

        h.Hex8Operator._launch = launch
        h.Hex8Operator._action_partial_capacity = lambda op: (op.n + 127) // 128

    if variant not in ("krylov_separate", "spatial_rho"):
        return

    def spatial_beta(pcg, iteration, threshold, mode):
        fft = pcg.fused
        if not hasattr(pcg, "_experimental_rho"):
            pcg._experimental_rho = cp.empty(h._reduction_grid(pcg.n3), cp.float64)
        rho = pcg._experimental_rho
        h._reduction_kernel("hex8_dot_partial", pcg.r.dtype, pcg.z.dtype)(
            (rho.size,), (h._REDUCTION_THREADS,), (pcg.r, pcg.z, rho, np.int64(pcg.n3))
        )
        pcg.k["hex8_pcg_beta"](
            (1,),
            (h._REDUCTION_THREADS,),
            (
                rho,
                np.int32(rho.size),
                fft.rmax,
                np.int32(fft.rmax.size),
                pcg.s,
                pcg.state,
                np.float64(threshold),
                np.int32(iteration),
                np.int32(mode),
                pcg.replace,
                pcg.history,
                pcg.gamma,
                pcg.energy,
            ),
        )
        counts["spatial_reductions"] += 1

    if variant == "spatial_rho":

        def prepare_spatial(pcg, iteration, threshold, mode):
            fft = pcg.fused
            fft.forward(pcg.r, pcg.state)
            fft.multiply(pcg.state)
            fft.backward(pcg.z, fft.idle)
            spatial_beta(pcg, iteration, threshold, mode)

        f.FusedCudaPCG._precondition = prepare_spatial

    def iterate(pcg, x, first, last, threshold):
        k, fft = pcg.k, pcg.fused
        x = pcg.increment if pcg.mixed else x
        if not isinstance(x, cp.ndarray):
            raise RuntimeError(
                "experimental separated Krylov requires device-resident iterate"
            )
        if not hasattr(pcg, "_experimental_rmax"):
            pcg._experimental_rmax = cp.empty_like(fft.rmax)
        polled = None
        for iteration in range(first + 1, last + 1):
            m = pcg.op._action_partials(pcg.p, pcg.ap, pcg.state, pcg.pap)
            k["hex8_pcg_alpha"](
                (1,),
                (h._REDUCTION_THREADS,),
                (
                    pcg.pap,
                    np.int32(m),
                    pcg.s,
                    pcg.state,
                    pcg.history,
                    np.int32(iteration),
                ),
            )
            # A separate full vector pass computes x, r and max|r|.
            k["hex8_pcg_update"](
                (fft.rmax.size,),
                (h._REDUCTION_THREADS,),
                (
                    pcg.p,
                    pcg.ap,
                    x,
                    pcg.r,
                    pcg.s,
                    pcg.state,
                    pcg._experimental_rmax,
                    np.int64(pcg.n3),
                ),
            )
            fft.forward(pcg.r, pcg.state)  # generated FFT, no fused update
            cp.copyto(fft.rmax, pcg._experimental_rmax)
            fft.multiply(pcg.state)
            if variant == "spatial_rho":
                fft.backward(pcg.z, fft.idle)
                spatial_beta(pcg, iteration, threshold, 0)
            else:
                pcg._beta(iteration, threshold, 0)
                fft.backward(pcg.z, fft.idle)
            k["hex8_pcg_direction"](
                (pcg.grid,),
                (h._REDUCTION_THREADS,),
                (pcg.z, pcg.p, pcg.s, pcg.state, np.int64(pcg.n3)),
            )
            counts["separate_iterations"] += 1
            slot = iteration % 2
            pcg.state.get(out=pcg.host[slot], blocking=False)
            pcg.events[slot].record()
            if polled is not None:
                pcg.events[polled % 2].synchronize()
                if pcg.host[polled % 2, 0]:
                    break
            polled = iteration
        state = pcg.state.get()
        return int(state[0]), int(state[1])

    f.FusedCudaPCG.iterate = iterate


def patch_gani_partial(variant, counts):
    """Separate pending GaNi vector work while retaining its generated FFT."""
    import cupy as cp
    import numpy as np
    from cufhom.fft.linear import common, coordinates, fused

    # This disables a small-grid-only grouped prefix in every pilot arm; the
    # 189^3 production path does not use the prefix in the first place.
    coordinates._DEVICE_PREFIX_MAX_VOXELS = 0
    if variant == "identity_reference":
        original = common.isotropic_reference

        def reference(stiffness, reference=None):
            if reference is not None:
                return original(stiffness, reference)
            _, _, shear = original(stiffness)
            C0 = 2 * shear * np.eye(6)
            counts["identity_reference_calls"] += 1
            counts["reference_matrix"] = C0.tolist()
            return original(stiffness, C0)

        common.isotropic_reference = reference
    if variant == "krylov_partial":
        original = fused.FusedPipeline.apply
        direction_kernel = cp.ElementwiseKernel(
            "T residual, T beta",
            "T p",
            "p = residual + beta * p",
            "public_gani_separate_direction",
        )

        def apply(pipeline, p, direction=None):
            had_pending = pipeline.pending is not None
            pipeline.settle()
            counts["separate_solution_updates"] += int(had_pending)
            if direction is not None:
                residual, beta = direction
                real = np.float32 if p.dtype == cp.complex64 else np.float64
                direction_kernel(residual.view(real), real(beta), p.view(real))
                counts["separate_direction_updates"] += 1
            return original(pipeline, p, None)

        fused.FusedPipeline.apply = apply


def patch_gani_vectors(counts):
    """Separate GaNi solution, residual and direction updates from its FFT."""
    import re

    import cupy as cp
    import numpy as np
    from cufhom.fft.linear import coordinates, fused

    patch_gani_partial("krylov_partial", counts)
    residual_update = cp.ElementwiseKernel(
        "T ap, raw T alpha",
        "T residual",
        "residual -= alpha[0] * ap;",
        "public_gani_separate_residual",
    )
    original_init = fused.FusedPipeline.__init__

    def initialize(pipeline, *args, **kwargs):
        original_init(pipeline, *args, **kwargs)
        source, entry, smem = pipeline._parts["specs"]["k_afwd"]
        marker = "// Issue the contiguous residual loads first;"
        start = source.index(marker)
        twiddle = re.search(
            r" for\(int i=tid;i<\d+;i\+=\d+\)tw\[i\]=tw_g\[i\];", source[start:]
        )
        if not twiddle:
            raise RuntimeError("Generated FFT source changed: twiddle loop not found")
        source = source[:start] + source[start + twiddle.start() :]
        start = source.index(" // A failed energy check poisons")
        end = source.index(" #pragma unroll", start)
        source = source[:start] + source[end:]
        match = re.search(
            r"     (?:float2|double2) ri=rv\[q\]\[c\];.*?(?=\n   \}\n \}\n)",
            source,
            flags=re.S,
        )
        if not match:
            raise RuntimeError(
                "Generated FFT source changed: residual update not found"
            )
        source = source[: match.start()] + "     r[o+c]=ap[c];" + source[match.end() :]
        tail = source.index("\n   }\n }\n", match.start()) + len("\n   }\n }\n")
        source = source[:tail] + "}\n"
        pipeline._experimental_afwd = fused._compile(source, entry, smem)
        pipeline._experimental_action = None
        pipeline._experimental_partial = None
        counts["compiled_separate_fft_kernels"] += 1

    def update(pipeline, p, numerator, denominator, x, r, smoothing=None):
        if not isinstance(x, cp.ndarray):
            raise RuntimeError(
                "experimental separated update requires device-resident iterate"
            )
        pipeline.settle()
        if pipeline._experimental_action is None:
            pipeline._experimental_action = cp.empty_like(r)
            pipeline._experimental_partial = cp.empty(
                coordinates.reduction_blocks(r.size), cp.float64
            )
        ap = pipeline._experimental_action
        pipeline._experimental_afwd(
            pipeline.grid_a,
            (pipeline.config["nta_fwd"],),
            (
                pipeline.S,
                ap,
                np.float64(1.0),
                pipeline.unit_energy,
                pipeline.step,
                pipeline.partial_a,
                pipeline.tw0,
                pipeline.counters[1:2],
                pipeline.norm,
            )
            + pipeline.moduli,
            shared_mem=pipeline.smem_a,
        )
        pipeline.generic["step"](
            (1,), (1,), (np.float64(numerator), denominator, pipeline.step)
        )
        y, blend = (None, 0.0) if smoothing is None else smoothing
        pipeline.pending = (p, x, y, blend)
        pipeline.settle()
        counts["separate_solution_updates"] += 1
        real = np.float32 if r.dtype == cp.complex64 else np.float64
        residual_update(ap.view(real), pipeline.step, r.view(real))
        partial = pipeline._experimental_partial
        dot = coordinates._mixed_kernel(
            "mixed_dot" if r.dtype == cp.complex64 else "mixed_dot128"
        )
        dot((partial.size,), (256,), (r, r, partial, np.int64(r.size)))
        counts["separate_residual_updates"] += 1
        return float(cp.sum(partial, dtype=cp.float64))

    fused.FusedPipeline.__init__ = initialize
    fused.FusedPipeline.update = update


def patch_gani_energy(counts):
    """Separate the GaNi energy reduction as well as its vector updates."""
    import re

    import cupy as cp
    import numpy as np
    from cufhom.fft.linear import coordinates, fused

    patch_gani_vectors(counts)
    init0 = fused.FusedPipeline.__init__
    apply0 = fused.FusedPipeline.apply
    residual_update = cp.ElementwiseKernel(
        "T ap, raw T alpha",
        "T residual",
        "residual -= alpha[0]*ap;",
        "public_gani_separate_residual_energy",
    )

    def initialize(pipeline, *args, **kwargs):
        init0(pipeline, *args, **kwargs)
        source, entry, smem = pipeline._parts["specs"]["k_c"]
        source, count = re.subn(
            r"\s*(?:float|double) ev=0\.(?:f)?;\s*for\(int c=0;c<6;c\+\+\)ev=(?:fmaf|fma)\(x\[c\],v\[e\]\[c\],ev\);\s*sum\+=ev;",
            "",
            source,
        )
        if count != 1:
            raise RuntimeError(
                f"expected exactly one material energy accumulation, got {count}"
            )
        reduce = fused.cuda_source(fused._REDUCE, pipeline.dtype)
        if source.count(reduce) != 1:
            raise RuntimeError("KC reduction source changed")
        source = source.replace(reduce, "")
        pipeline.k_c = fused._compile(source, entry, smem)
        counts["compiled_separate_energy_kernels"] += 1

    def apply(pipeline, p, direction=None):
        apply0(pipeline, p, direction)
        if pipeline._experimental_action is None:
            pipeline._experimental_action = cp.empty_like(p)
            pipeline._experimental_partial = cp.empty(
                coordinates.reduction_blocks(p.size), cp.float64
            )
        hp = pipeline._experimental_action
        pipeline._experimental_afwd(
            pipeline.grid_a,
            (pipeline.config["nta_fwd"],),
            (
                pipeline.S,
                hp,
                np.float64(1.0),
                pipeline.unit_energy,
                pipeline.step,
                pipeline.partial_a,
                pipeline.tw0,
                pipeline.counters[1:2],
                pipeline.norm,
            )
            + pipeline.moduli,
            shared_mem=pipeline.smem_a,
        )
        partial = pipeline._experimental_partial
        dot = coordinates._mixed_kernel(
            "mixed_dot" if p.dtype == cp.complex64 else "mixed_dot128"
        )
        dot((partial.size,), (256,), (p, hp, partial, np.int64(p.size)))
        counts["separate_energy_reductions"] += 1
        return cp.sum(partial, dtype=cp.float64)

    def update(pipeline, p, numerator, denominator, x, r, smoothing=None):
        if not isinstance(x, cp.ndarray):
            raise RuntimeError("device-resident iterate required")
        pipeline.settle()
        hp = pipeline._experimental_action
        pipeline.generic["step"](
            (1,), (1,), (np.float64(numerator), denominator, pipeline.step)
        )
        y, blend = (None, 0.0) if smoothing is None else smoothing
        pipeline.pending = (p, x, y, blend)
        pipeline.settle()
        counts["separate_solution_updates"] += 1
        real = np.float32 if r.dtype == cp.complex64 else np.float64
        residual_update(hp.view(real), pipeline.step, r.view(real))
        partial = pipeline._experimental_partial
        dot = coordinates._mixed_kernel(
            "mixed_dot" if r.dtype == cp.complex64 else "mixed_dot128"
        )
        dot((partial.size,), (256,), (r, r, partial, np.int64(r.size)))
        counts["separate_residual_updates"] += 1
        return float(cp.sum(partial, dtype=cp.float64))

    fused.FusedPipeline.__init__ = initialize
    fused.FusedPipeline.apply = apply
    fused.FusedPipeline.update = update


def patch_gani_slabs(counts):
    """Reorder generated GaNi slabs into whole-grid stages without changing FFTs."""
    from cufhom.fft.linear import fused

    def staged(self, done=None):
        cfg = self.config
        n0, n1, _ = self.shape
        chunk, lpb = cfg["slabs"], cfg["lpb"]
        groups = -(-n0 * n1 // lpb)
        material = self._material()
        done = self.never_done if done is None else done
        slabs = [(s0, min(chunk, n0 - s0)) for s0 in range(0, n0, chunk)]
        for s0, c in slabs:
            self.k_binv(
                (self.grid_b[0], c),
                (cfg["ntb"],),
                (self.S, self.tw1, np.int32(s0), done),
                shared_mem=self.smem_b,
            )
        for s0, c in slabs:
            self.k_c(
                (-(-c * n1 // lpb),),
                (cfg["ntc"],),
                (self.S,)
                + material
                + (
                    self.partial_c,
                    self.tw2,
                    self.twn,
                    self.counters[0:1],
                    self.energy,
                    np.int64(s0 * n1 // lpb),
                    np.uint32(groups),
                    done,
                ),
                shared_mem=self.smem_c,
            )
        for s0, c in slabs:
            self.k_bfwd(
                (self.grid_b[0], c),
                (cfg["ntb"],),
                (self.S, self.tw1, np.int32(s0), done),
                shared_mem=self.smem_b,
            )
        counts["staged_actions"] += 1
        counts["staged_slabs"] += len(slabs)

    fused.FusedPipeline._slab_passes = staged
