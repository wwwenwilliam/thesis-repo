import time
import cupy as cp
import numpy as np
from cuvs.neighbors import ivf_flat


def _gpu_mem_used_mb():
    free, total = cp.cuda.Device(0).mem_info
    return (total - free) / 1e6


def ivf_threshold_join(vectors_A, vectors_B, threshold, n_lists, n_probes, k_candidates, batch_size, self_join, prebuild_blocks=True, return_chunks=False):
    """IVF-Flat based similarity join with threshold using blocked indices.
    
    Args:
        prebuild_blocks: If True, build all IVF indices upfront (faster but uses more GPU memory).
                         If False, build each index on-the-fly during search (slower but fits in VRAM).
    """
    print(f"  params: n_lists={n_lists}, n_probes={n_probes}, k_candidates={k_candidates}, batch_size={batch_size:,}, prebuild={prebuild_blocks}")
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    
    # Accumulate results on GPU to avoid per-tile sync from .get()
    gpu_results_a = []
    gpu_results_b = []
    gpu_results_dist = []
    
    # Instrumentation
    total_pairs_compared = 0
    total_searches = 0
    peak_mem = _gpu_mem_used_mb()
    
    t0 = time.perf_counter()
    build_params = ivf_flat.IndexParams(n_lists=n_lists)
    search_params = ivf_flat.SearchParams(n_probes=n_probes)
    
    # Pre-build mode: build all indices upfront
    if prebuild_blocks:
        db_blocks = []
        for a_start in range(0, N, batch_size):
            a_end = min(a_start + batch_size, N)
            db_batch = cp.asarray(vectors_A[a_start:a_end])
            cp.get_default_memory_pool().free_all_blocks()
            index = ivf_flat.build(build_params, db_batch)
            db_blocks.append((a_start, a_end, index))
        
        cp.cuda.Stream.null.synchronize()
        build_time = time.perf_counter() - t0
        print(f"IVF build ({len(db_blocks)} blocks): {build_time:.4f}s")
    
    # Cross-search
    t_search = 0.0
    t_filter = 0.0
    t_build_inline = 0.0
    t1 = time.perf_counter()
    
    # Generate DB block ranges
    db_ranges = [(a_s, min(a_s + batch_size, N)) for a_s in range(0, N, batch_size)]
    if not prebuild_blocks:
        n_blocks = len(db_ranges)
        print(f"IVF on-the-fly build ({n_blocks} blocks)")
    
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        q_batch = cp.asarray(vectors_B[b_start:b_end])
        n_queries = b_end - b_start
        
        for block_idx, (a_start, a_end) in enumerate(db_ranges):
            # Self-join triangle skip
            if self_join and a_start < b_start:
                continue
            
            # Get or build the index
            if prebuild_blocks:
                index = db_blocks[block_idx][2]
            else:
                tb = time.perf_counter()
                db_batch = cp.asarray(vectors_A[a_start:a_end])
                cp.get_default_memory_pool().free_all_blocks()
                index = ivf_flat.build(build_params, db_batch)
                cp.cuda.Stream.null.synchronize()
                t_build_inline += time.perf_counter() - tb
            
            ts = time.perf_counter()
            cp.get_default_memory_pool().free_all_blocks()
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
                mask &= (cp.arange(b_start, b_end)[:, None] < neighbors)
            
            b_local, k_local = cp.where(mask)
            if b_local.size > 0:
                gpu_results_a.append(neighbors[b_local, k_local])
                gpu_results_b.append(b_local + b_start)
                gpu_results_dist.append(distances[b_local, k_local])
            cp.cuda.Stream.null.synchronize()
            t_filter += time.perf_counter() - tf
            
            # Free on-the-fly index AFTER results are extracted
            if not prebuild_blocks:
                del index, db_batch, distances, neighbors
                cp.get_default_memory_pool().free_all_blocks()
    
    cp.cuda.Stream.null.synchronize()
    peak_mem = max(peak_mem, _gpu_mem_used_mb())
    total_time = time.perf_counter() - t1
    print(f"IVF search+filter: {total_time:.4f}s")
    build_str = f" | inline build: {t_build_inline:.2f}s" if not prebuild_blocks else ""
    print(f"  search: {t_search:.2f}s | filter: {t_filter:.2f}s{build_str} | {total_searches} searches")
    print(f"  pairs compared: {total_pairs_compared:,} ({total_pairs_compared/1e9:.2f}B)")
    print(f"  peak GPU memory: {peak_mem:.0f} MB")
    
    if return_chunks:
        return [g.get() for g in gpu_results_a], [g.get() for g in gpu_results_b], [g.get() for g in gpu_results_dist]
    
    if not gpu_results_a:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    
    # Single bulk transfer to CPU
    all_a = cp.concatenate(gpu_results_a).get()
    all_b = cp.concatenate(gpu_results_b).get()
    all_dist = cp.concatenate(gpu_results_dist).get()
    
    return all_a, all_b, all_dist
