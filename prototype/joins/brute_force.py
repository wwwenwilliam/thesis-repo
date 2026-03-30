"""
Brute force threshold similarity join on GPU.

Takes two numpy batches (single tile), uploads to GPU, computes pairwise
L2 distances, filters by threshold, and returns matching pairs as numpy.
"""
import cupy as cp
import numpy as np


@cp.fuse()
def _fused_l2(dot, q_norms, a_norms):
    """Fused L2 distance: ||q - a||^2 = ||q||^2 - 2*q·a + ||a||^2"""
    d = q_norms - 2.0 * dot + a_norms
    return cp.maximum(d, 0.0)


def brute_force_join(chunk_A, chunk_B, threshold, self_join_diagonal=False, params=None):
    """Compute threshold join between two vector batches.

    This is a pure compute function for a single tile. The batching layer
    is responsible for iterating over tile pairs and writing results to disk.

    Args:
        chunk_A: numpy float32 array, shape (nA, d) — "database" side vectors.
        chunk_B: numpy float32 array, shape (nB, d) — "query" side vectors.
        threshold: L2 distance threshold (squared).
        self_join_diagonal: If True, this tile is on the diagonal of a
            self-join, so apply upper-triangle masking (only keep pairs
            where local_a > local_b) to avoid duplicates and self-matches.

    Returns:
        (a_indices, b_indices, distances) — numpy arrays of local indices
        within the chunk and their L2 distances. Empty arrays if no matches.
    """
    gpu_B = cp.asarray(chunk_B)
    b_norms = cp.sum(gpu_B ** 2, axis=1)[:, None]

    if self_join_diagonal:
        gpu_A = gpu_B
        a_norms = b_norms.T
    else:
        gpu_A = cp.asarray(chunk_A)
        a_norms = cp.sum(gpu_A ** 2, axis=1)[None, :]

    dists = _fused_l2(cp.dot(gpu_B, gpu_A.T), b_norms, a_norms)

    mask = dists <= threshold

    if self_join_diagonal:
        mask = cp.triu(mask, k=1)

    b_local, a_local = cp.where(mask)

    if b_local.size == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    result_dists = dists[b_local, a_local]

    nA = gpu_A.shape[1] if self_join_diagonal else gpu_A.shape[0]
    nB = gpu_B.shape[0]
    if self_join_diagonal:
        # N * (N - 1) / 2 for self-join upper triangle
        pairs_compared = nB * (nB - 1) // 2
    else:
        pairs_compared = nA * nB

    extra_stats = {"pairs_compared": int(pairs_compared)}

    return (a_local.get().astype(np.int64), 
            b_local.get().astype(np.int64), 
            result_dists.get().astype(np.float32),
            extra_stats)
