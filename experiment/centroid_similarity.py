import time
import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force
from cuvs.cluster import kmeans


def run_threshold_similarity_join(vectors_A, vectors_B, threshold, n_clusters, k_db_candidates, batch_size, self_join):
    """
    Centroid-based similarity join with threshold.
    
    For each (D_batch, Q_batch) pair:
        1. Build k-NN index on D_batch
        2. Partition Q_batch with k-means
        3. For each centroid, do k-NN query on D_batch index
        4. Brute force filter pairs between k-NN result and cluster members
    """
    N, M = vectors_A.shape[0], vectors_B.shape[0]
    results = []
    
    t0 = time.perf_counter()
    for a_start in range(0, N, batch_size):
        a_end = min(a_start + batch_size, N)
        db_batch = vectors_A[a_start:a_end]
        d_index = brute_force.build(db_batch)
        
        for b_start in range(0, M, batch_size):
            b_end = min(b_start + batch_size, M)
            q_batch = vectors_B[b_start:b_end]
            
            # Cluster Q_batch
            cluster_params = kmeans.KMeansParams(n_clusters=n_clusters)
            centroids, _, _ = kmeans.fit(cluster_params, q_batch)
            
            # Assign queries to centroids
            c_index = brute_force.build(centroids)
            _, labels = brute_force.search(c_index, q_batch, k=1)
            labels = cp.asarray(labels).flatten()
            
            # k-NN search for each centroid
            _, centroid_nn = brute_force.search(d_index, centroids, k=k_db_candidates)
            centroid_nn = cp.asarray(centroid_nn)
            
            # Brute force filter per cluster
            for c in range(n_clusters):
                members = cp.where(labels == c)[0]
                if members.size == 0:
                    continue
                
                q_indices = members + b_start
                q_vectors = q_batch[members]
                db_local = centroid_nn[c]
                db_global = db_local + a_start
                db_vectors = db_batch[db_local]
                
                # Squared L2
                q_sq = cp.sum(q_vectors ** 2, axis=1)
                db_sq = cp.sum(db_vectors ** 2, axis=1)
                sq_dists = q_sq[:, None] + db_sq[None, :] - 2 * (q_vectors @ db_vectors.T)
                sq_dists = cp.maximum(sq_dists, 0)
                
                mask = sq_dists <= threshold
                if self_join:
                    a_grid, b_grid = cp.meshgrid(db_global, q_indices)
                    mask &= (b_grid < a_grid)
                
                q_local, d_local = cp.where(mask)
                if q_local.size > 0:
                    results.append((
                        db_global[d_local].get(),
                        q_indices[q_local].get(),
                        sq_dists[q_local, d_local].get()
                    ))
    
    cp.cuda.Stream.null.synchronize()
    print(f"Centroid join: {time.perf_counter() - t0:.4f}s")
    
    if not results:
        return []
    all_a = np.concatenate([r[0] for r in results])
    all_b = np.concatenate([r[1] for r in results])
    all_dist = np.concatenate([r[2] for r in results])
    return list(zip(all_a.tolist(), all_b.tolist(), all_dist.tolist()))
