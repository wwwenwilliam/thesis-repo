import time
import cupy as cp
import numpy as np
from cuvs.neighbors import ivf_flat


def _gpu_mem_used_mb():
    free, total = cp.cuda.Device(0).mem_info
    return (total - free) / 1e6


def ivf_threshold_join(vectors_A, vectors_B, threshold, n_lists, n_probes, k_candidates, batch_size, self_join):
    """IVF-Flat based similarity join with threshold using blocked indices."""
    print(f"  params: n_lists={n_lists}, n_probes={n_probes}, k_candidates={k_candidates}, batch_size={batch_size:,}")
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    
    # Accumulate results on GPU to avoid per-tile sync from .get()
    gpu_results_a = []
    gpu_results_b = []
    gpu_results_dist = []
    
    # Instrumentation
    total_pairs_compared = 0
    total_searches = 0
    peak_mem = _gpu_mem_used_mb()
    
    # Build IVF indices for each database block
    t0 = time.perf_counter()
    build_params = ivf_flat.IndexParams(n_lists=n_lists)
    search_params = ivf_flat.SearchParams(n_probes=n_probes)
    
    db_blocks = []
    for a_start in range(0, N, batch_size):
        a_end = min(a_start + batch_size, N)
        db_batch = cp.asarray(vectors_A[a_start:a_end])
        cp.get_default_memory_pool().free_all_blocks()  # Yield CuPy cache to RMM
        index = ivf_flat.build(build_params, db_batch)
        db_blocks.append((a_start, a_end, index))
    
    cp.cuda.Stream.null.synchronize()
    build_time = time.perf_counter() - t0
    print(f"IVF build ({len(db_blocks)} blocks): {build_time:.4f}s")
    
    # Cross-search
    t_search = 0.0
    t_filter = 0.0
    t0 = time.perf_counter()
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        q_batch = cp.asarray(vectors_B[b_start:b_end])
        n_queries = b_end - b_start
        
        for a_start, a_end, index in db_blocks:
            # Self-join triangle skip: only search blocks where a_start >= b_start
            if self_join and a_start < b_start:
                continue
            
            ts = time.perf_counter()
            cp.get_default_memory_pool().free_all_blocks()  # Yield CuPy cache to RMM
            distances, neighbors = ivf_flat.search(search_params, index, q_batch, k=k_candidates)
            cp.cuda.Stream.null.synchronize()
            t_search += time.perf_counter() - ts
            
            tf = time.perf_counter()
            distances = cp.asarray(distances)
            neighbors = cp.asarray(neighbors) + a_start
            
            total_pairs_compared += n_queries * k_candidates
            total_searches += 1
            
            mask = distances <= threshold
            if self_join and a_start == b_start:
                # Only enforce a_idx > b_idx within diagonal blocks
                mask &= (cp.arange(b_start, b_end)[:, None] < neighbors)
            
            b_local, k_local = cp.where(mask)
            if b_local.size > 0:
                # Keep results on GPU; transfer to CPU in bulk at the end
                gpu_results_a.append(neighbors[b_local, k_local])
                gpu_results_b.append(b_local + b_start)
                gpu_results_dist.append(distances[b_local, k_local])
            cp.cuda.Stream.null.synchronize()
            t_filter += time.perf_counter() - tf
    
    cp.cuda.Stream.null.synchronize()
    peak_mem = max(peak_mem, _gpu_mem_used_mb())
    total_time = time.perf_counter() - t0
    print(f"IVF search+filter: {total_time:.4f}s")
    print(f"  search: {t_search:.2f}s | filter: {t_filter:.2f}s | {total_searches} searches")
    print(f"  pairs compared: {total_pairs_compared:,} ({total_pairs_compared/1e9:.2f}B)")
    print(f"  peak GPU memory: {peak_mem:.0f} MB")
    
    if not gpu_results_a:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    
    # Single bulk transfer to CPU
    all_a = cp.concatenate(gpu_results_a).get()
    all_b = cp.concatenate(gpu_results_b).get()
    all_dist = cp.concatenate(gpu_results_dist).get()
    
    return all_a, all_b, all_dist
