import time
import cupy as cp
import numpy as np


def brute_force_threshold_join(vectors_A, vectors_B, threshold, batch_size, self_join):
    """GPU brute force similarity join with threshold."""
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    results = []
    
    t0 = time.perf_counter()
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        q_batch = vectors_B[b_start:b_end]
        q_sq = cp.sum(q_batch ** 2, axis=1)
        
        for a_start in range(0, N, batch_size):
            a_end = min(a_start + batch_size, N)
            db_batch = vectors_A[a_start:a_end]
            
            # Squared L2: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            a_sq = cp.sum(db_batch ** 2, axis=1)
            sq_dists = q_sq[:, None] + a_sq[None, :] - 2 * (q_batch @ db_batch.T)
            sq_dists = cp.maximum(sq_dists, 0)
            
            mask = sq_dists <= threshold
            if self_join:
                a_grid, b_grid = cp.meshgrid(cp.arange(a_start, a_end), cp.arange(b_start, b_end))
                mask &= (b_grid < a_grid)
            
            b_local, a_local = cp.where(mask)
            if b_local.size > 0:
                results.append((
                    (a_local + a_start).get(),
                    (b_local + b_start).get(),
                    sq_dists[b_local, a_local].get()
                ))
    
    cp.cuda.Stream.null.synchronize()
    print(f"Brute force: {time.perf_counter() - t0:.4f}s")
    
    if not results:
        return []
    all_a = np.concatenate([r[0] for r in results])
    all_b = np.concatenate([r[1] for r in results])
    all_dist = np.concatenate([r[2] for r in results])
    return list(zip(all_a.tolist(), all_b.tolist(), all_dist.tolist()))
