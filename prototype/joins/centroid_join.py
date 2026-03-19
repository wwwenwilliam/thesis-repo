import time
import cupy as cp
from cupyx.profiler import time_range
import numpy as np
import rmm
from cuvs.cluster import kmeans
from cuvs.neighbors import brute_force

@cp.fuse()
def _fused_l2_dist(dot_product, q_norms, a_norms):
    """Fuse element-wise ops into single kernel."""
    d = q_norms - 2.0 * dot_product + a_norms
    return cp.maximum(d, 0.0)

def centroid_join(chunk_A, chunk_B, threshold, self_join_diagonal=False, n_clusters=1024, k_db_candidates=8192):
    """
    Centroid based similarity join logic.
    Run on a single pair of chunks (A and B).
    """
    if len(chunk_A) == 0 or len(chunk_B) == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                {})

    with time_range("centroid_join", color_id=0):
        # Prevent RMM pool from hoarding GPU memory (kmeans grows the pool
        # and never releases it, starving subsequent cudaMalloc calls)
        rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

        db_batch = cp.asarray(chunk_A)
        q_batch = cp.asarray(chunk_B)

        # Need at least as many points as clusters
        n_clusters = min(n_clusters, len(q_batch))

        # 1. Cluster chunk_B on a 10% random sample
        with time_range("centroid_join/kmeans_fit", color_id=1):
            cp.cuda.Stream.null.synchronize()
            t_km = time.time()

            n_samples = max(n_clusters, len(q_batch) // 10)
            if n_samples < len(q_batch):
                # Sample uniformly without replacement
                sample_indices = cp.random.choice(len(q_batch), size=n_samples, replace=False)
                q_sample = q_batch[sample_indices]
            else:
                q_sample = q_batch

            cluster_params = kmeans.KMeansParams(n_clusters=n_clusters)
            cp.get_default_memory_pool().free_all_blocks()
            centroids, _, _ = kmeans.fit(cluster_params, q_sample)

        with time_range("centroid_join/assign_queries", color_id=2):
            # Assign queries to centroids manually using brute force
            cp.get_default_memory_pool().free_all_blocks()
            c_index = brute_force.build(centroids)
            _, labels = brute_force.search(c_index, q_batch, k=1)
            labels = cp.asarray(labels).flatten()
            cp.cuda.Stream.null.synchronize()
            t_kmeans_total = time.time() - t_km

        # 2. Build index on chunk_A using brute force again for exact centroid matching
        with time_range("centroid_join/build_db_index", color_id=3):
            cp.cuda.Stream.null.synchronize()
            t_s = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            d_index = brute_force.build(db_batch, metric="sqeuclidean")

            # Search centroids against chunk_A
            cp.get_default_memory_pool().free_all_blocks()
            k_db = min(k_db_candidates, len(db_batch))
            _, centroid_nn = brute_force.search(d_index, centroids, k=k_db)
            centroid_nn = cp.asarray(centroid_nn)
            cp.cuda.Stream.null.synchronize()
            t_search_total = time.time() - t_s

        # 3. Brute force filter
        with time_range("centroid_join/filter_loop", color_id=4):
            t_bf = time.time()
            block_a = []
            block_b = []
            block_d = []
            total_filter_ops = 0

            for c in range(centroids.shape[0]):
                members = cp.where(labels == c)[0]
                if members.size == 0:
                    continue

                q_indices = members
                q_vectors = q_batch[members]
                db_local = centroid_nn[c]
                db_vectors = db_batch[db_local]

                # Count direct comparison pairs
                total_filter_ops += q_vectors.shape[0] * db_vectors.shape[0]

                q_sq = cp.sum(q_vectors ** 2, axis=1)
                db_sq = cp.sum(db_vectors ** 2, axis=1)

                cp.get_default_memory_pool().free_all_blocks()
                sq_dists = _fused_l2_dist(q_vectors @ db_vectors.T, q_sq[:, None], db_sq[None, :])

                mask = sq_dists <= threshold
                if self_join_diagonal:
                    # a_local > b_local for self join upper triangle
                    mask &= (q_indices[:, None] < db_local[None, :])

                q_local_idx, d_local_idx = cp.where(mask)
                if q_local_idx.size > 0:
                    block_a.append(db_local[d_local_idx])
                    block_b.append(q_indices[q_local_idx])
                    block_d.append(sq_dists[q_local_idx, d_local_idx])

            cp.cuda.Stream.null.synchronize()
            t_bruteforce_total = time.time() - t_bf

        with time_range("centroid_join/collect_results", color_id=5):
            cpu_a = cp.concatenate(block_a).get().astype(np.int64)
            cpu_b = cp.concatenate(block_b).get().astype(np.int64)
            cpu_d = cp.concatenate(block_d).get().astype(np.float32)

        extra_stats = {
            "build_time_s": t_kmeans_total + t_search_total,
            "pairs_compared": int(total_filter_ops)
        }

        return cpu_a, cpu_b, cpu_d, extra_stats
