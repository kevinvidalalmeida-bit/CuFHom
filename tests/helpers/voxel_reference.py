import numpy as np

def _voxelized_a2_reference(phase: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    phase_flat = np.asarray(phase).reshape(-1)
    orientation_flat = np.asarray(orientations).reshape(-1, 3)
    moment_sum = np.zeros((3, 3), dtype=float)
    valid_count = 0
    chunk_size = 1_000_000
    for start in range(0, len(phase_flat), chunk_size):
        stop = min(len(phase_flat), start + chunk_size)
        mask = phase_flat[start:stop] != 0
        if not np.any(mask):
            continue
        directions = np.asarray(
            orientation_flat[start:stop][mask],
            dtype=float,
        )
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            continue
        directions = directions[valid] / norms[valid, None]
        moment_sum += np.einsum("ni,nj->ij", directions, directions)
        valid_count += len(directions)
    if valid_count == 0:
        return np.zeros((3, 3), dtype=float)
    return moment_sum / valid_count
