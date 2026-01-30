import time
import cupy as cp
import numpy as np
from cuvs.neighbors import ivf_flat


def ivf_threshold_join(vectors_A, vectors_B, threshold, n_lists, n_probes, k_candidates, batch_size, self_join):
    """IVF-Flat based similarity join with threshold using blocked indices."""
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    results = []
    
    # Build IVF indices for each database block
    t0 = time.perf_counter()
    build_params = ivf_flat.IndexParams(n_lists=n_lists)
    search_params = ivf_flat.SearchParams(n_probes=n_probes)
    
    db_blocks = []
    for a_start in range(0, N, batch_size):
        a_end = min(a_start + batch_size, N)
        index = ivf_flat.build(build_params, vectors_A[a_start:a_end])
        db_blocks.append((a_start, a_end, index))
    
    cp.cuda.Stream.null.synchronize()
    print(f"IVF build ({len(db_blocks)} blocks): {time.perf_counter() - t0:.4f}s")
    
    # Cross-search
    t0 = time.perf_counter()
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        q_batch = vectors_B[b_start:b_end]
        
        for a_start, a_end, index in db_blocks:
            distances, neighbors = ivf_flat.search(search_params, index, q_batch, k=k_candidates)
            distances = cp.asarray(distances)
            neighbors = cp.asarray(neighbors) + a_start
            
            mask = distances <= threshold
            if self_join:
                mask &= (cp.arange(b_start, b_end)[:, None] < neighbors)
            
            b_local, k_local = cp.where(mask)
            if b_local.size > 0:
                results.append((
                    neighbors[b_local, k_local].get(),
                    (b_local + b_start).get(),
                    distances[b_local, k_local].get()
                ))
    
    cp.cuda.Stream.null.synchronize()
    print(f"IVF search+filter: {time.perf_counter() - t0:.4f}s")
    
    if not results:
        return []
    all_a = np.concatenate([r[0] for r in results])
    all_b = np.concatenate([r[1] for r in results])
    all_dist = np.concatenate([r[2] for r in results])
    return list(zip(all_a.tolist(), all_b.tolist(), all_dist.tolist()))
