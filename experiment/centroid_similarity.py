import time
import cupy as cp
import numpy as np
import rmm
from cuvs.neighbors import brute_force
from cuvs.cluster import kmeans


def _gpu_mem_used_mb():
    free, total = cp.cuda.Device(0).mem_info
    return (total - free) / 1e6


@cp.fuse()
def _fused_l2_dist(dot_product, q_norms, a_norms):
    """Fuse element-wise ops into single kernel."""
    d = q_norms - 2.0 * dot_product + a_norms
    return cp.maximum(d, 0.0)


def run_threshold_similarity_join(vectors_A, vectors_B, threshold, n_clusters, k_db_candidates, batch_size, self_join, return_chunks=False):
    """
    Centroid-based similarity join with threshold.
    
    For each (D_batch, Q_batch) pair:
        1. Build k-NN index on D_batch
        2. Partition Q_batch with k-means
        3. For each centroid, do k-NN query on D_batch index
        4. Brute force filter pairs between k-NN result and cluster members
    """
    print(f"  params: n_clusters={n_clusters}, k_db_candidates={k_db_candidates}, batch_size={batch_size:,}")
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    
    # Prevent RMM pool from hoarding GPU memory (kmeans grows the pool
    # and never releases it, starving subsequent cudaMalloc calls)
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    
    # Accumulate results on CPU to avoid GPU memory buildup at scale
    cpu_results_a = []
    cpu_results_b = []
    cpu_results_dist = []
    
    # Instrumentation
    total_pairs_compared = 0
    total_cluster_iters = 0
    t_kmeans_total = 0.0
    t_search_total = 0.0
    t_bruteforce_total = 0.0
    n_block_pairs = 0
    peak_mem = _gpu_mem_used_mb()
    
    t0 = time.perf_counter()
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        q_batch = cp.asarray(vectors_B[b_start:b_end])
        
        # Cluster Q_batch just ONCE per query block
        t_km = time.perf_counter()
        cluster_params = kmeans.KMeansParams(n_clusters=n_clusters)
        cp.get_default_memory_pool().free_all_blocks()  # Yield CuPy cache to RMM
        centroids, _, _ = kmeans.fit(cluster_params, q_batch)
        
        # Assign queries to centroids
        cp.get_default_memory_pool().free_all_blocks()
        c_index = brute_force.build(centroids)
        _, labels = brute_force.search(c_index, q_batch, k=1)
        labels = cp.asarray(labels).flatten()
        cp.cuda.Stream.null.synchronize()
        t_kmeans_total += time.perf_counter() - t_km
        
        # Self-join triangle skip: DB blocks >= Query blocks
        a_start_idx = b_start if self_join else 0
        
        for a_start in range(a_start_idx, N, batch_size):
            a_end = min(a_start + batch_size, N)
            db_batch = cp.asarray(vectors_A[a_start:a_end])
            
            cp.get_default_memory_pool().free_all_blocks()  # Yield CuPy cache to RMM
            d_index = brute_force.build(db_batch)
            n_block_pairs += 1
            
            # k-NN search for each centroid against THIS database block
            t_s = time.perf_counter()
            cp.get_default_memory_pool().free_all_blocks()  # Yield CuPy cache to RMM
            _, centroid_nn = brute_force.search(d_index, centroids, k=k_db_candidates)
            centroid_nn = cp.asarray(centroid_nn)
            cp.cuda.Stream.null.synchronize()
            t_search_total += time.perf_counter() - t_s
            
            # Brute force filter per cluster
            block_a = []
            block_b = []
            block_d = []
            
            t_bf = time.perf_counter()
            for c in range(n_clusters):
                members = cp.where(labels == c)[0]
                if members.size == 0:
                    continue
                
                total_cluster_iters += 1
                q_indices = members + b_start
                q_vectors = q_batch[members]
                db_local = centroid_nn[c]
                db_global = db_local + a_start
                db_vectors = db_batch[db_local]
                
                n_members = int(members.size)
                n_db = int(db_local.size)
                total_pairs_compared += n_members * n_db
                
                # Fused squared L2 distance
                q_sq = cp.sum(q_vectors ** 2, axis=1)
                db_sq = cp.sum(db_vectors ** 2, axis=1)
                
                # Clear CuPy cache before large inner assignment
                cp.get_default_memory_pool().free_all_blocks()
                sq_dists = _fused_l2_dist(q_vectors @ db_vectors.T, q_sq[:, None], db_sq[None, :])
                
                mask = sq_dists <= threshold
                if self_join and b_start == a_start:
                    a_grid, b_grid = cp.meshgrid(db_global, q_indices)
                    mask &= (b_grid < a_grid)
                
                q_local, d_local = cp.where(mask)
                if q_local.size > 0:
                    block_a.append(db_global[d_local])
                    block_b.append(q_indices[q_local])
                    block_d.append(sq_dists[q_local, d_local])
            
            cp.cuda.Stream.null.synchronize()
            t_bruteforce_total += time.perf_counter() - t_bf
            
            # Batch transfer per block-pair
            if block_a:
                cpu_results_a.append(cp.concatenate(block_a).get())
                cpu_results_b.append(cp.concatenate(block_b).get())
                cpu_results_dist.append(cp.concatenate(block_d).get())
    
    cp.cuda.Stream.null.synchronize()
    peak_mem = max(peak_mem, _gpu_mem_used_mb())
    total_time = time.perf_counter() - t0
    print(f"Centroid join: {total_time:.4f}s")
    print(f"  kmeans+assign: {t_kmeans_total:.2f}s | centroid search: {t_search_total:.2f}s | brute force: {t_bruteforce_total:.2f}s")
    print(f"  {n_block_pairs} block pairs, {total_cluster_iters} cluster iterations")
    print(f"  pairs compared: {total_pairs_compared:,} ({total_pairs_compared/1e9:.2f}B)")
    print(f"  peak GPU memory: {peak_mem:.0f} MB")
    
    if return_chunks:
        return cpu_results_a, cpu_results_b, cpu_results_dist
    
    if not cpu_results_a:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    
    all_a = np.concatenate(cpu_results_a)
    all_b = np.concatenate(cpu_results_b)
    all_dist = np.concatenate(cpu_results_dist)
    
    return all_a, all_b, all_dist
