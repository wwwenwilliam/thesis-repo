"""
Self-Join Top-K Search Methods Comparison on SIFT1M.

Three methods compared:
1. Centroid-Restricted Search (clustering + local search)
2. IVF-Flat Search (standard approximate nearest neighbor)
3. Brute Force Search (ground truth)
"""
import time
import cupy as cp
from cuvs.neighbors import brute_force, ivf_flat
from cuvs.cluster import kmeans


def read_fvecs(filename: str) -> cp.ndarray:
    """Read fvecs format (standard for SIFT)."""
    import numpy as np
    with open(filename, 'rb') as f:
        f.read(4)  # skip first dim header
        f.seek(0)
        x = np.fromfile(f, dtype='float32')
        dim = 128  # SIFT dim
        vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
        print(f"Loaded {vectors.shape[0]:,} vectors from {filename}")
        return cp.asarray(vectors)


def centroid_search(
    vectors_A: cp.ndarray,
    vectors_B: cp.ndarray,
    k_global: int = 10,
    n_clusters: int = 1024,
    n_probes: int = 1,
    k_db_candidates: int = 25000,
    exclude_self: bool = True,
) -> tuple[list, float]:
    """
    Centroid-restricted search: cluster queries, search locally per cluster.
    
    Returns:
        results: List of (b_idx, a_idx, distance) tuples
        elapsed: Total time in seconds
    """
    t_start = time.perf_counter()
    M = vectors_B.shape[0]
    
    # Build index on A
    d_index = brute_force.build(vectors_A)
    
    # Cluster B using subset
    subset_size = min(M, 50_000)
    subset_idx = cp.random.choice(M, subset_size, replace=False)
    cluster_params = kmeans.KMeansParams(n_clusters=n_clusters)
    centroids, _, _ = kmeans.fit(cluster_params, vectors_B[subset_idx])
    
    # Multi-probe assignment
    c_index = brute_force.build(centroids)
    _, labels = brute_force.search(c_index, vectors_B, k=n_probes)
    
    # Flatten and sort by cluster
    labels_flat = cp.asarray(labels).flatten()
    query_indices = cp.arange(M, dtype=cp.int32)
    query_indices_flat = cp.repeat(query_indices, n_probes)
    
    sort_idx = cp.argsort(labels_flat)
    labels_sorted = labels_flat[sort_idx]
    query_indices_sorted = query_indices_flat[sort_idx]
    boundaries = cp.searchsorted(labels_sorted, cp.arange(n_clusters + 1))
    
    # Find database candidates per centroid
    _, topcentroidk_indices = brute_force.search(d_index, centroids, k=k_db_candidates)
    topcentroidk_indices = cp.asarray(topcentroidk_indices)
    
    # Per-cluster search with running top-K
    best_dists = cp.full(k_global, cp.inf)
    best_a = cp.zeros(k_global, dtype=cp.int32)
    best_b = cp.zeros(k_global, dtype=cp.int32)
    
    for i in range(n_clusters):
        start, end = int(boundaries[i]), int(boundaries[i+1])
        if start == end:
            continue
        
        current_q_indices = query_indices_sorted[start:end]
        cluster_queries = vectors_B[current_q_indices]
        
        db_subset_indices = topcentroidk_indices[i]
        db_subset_vectors = vectors_A[db_subset_indices]
        
        s_index = brute_force.build(db_subset_vectors)
        k_local = min(100, db_subset_vectors.shape[0])  # Increased for larger k_global
        dists, neighbors = brute_force.search(s_index, cluster_queries, k=k_local)
        
        dists = cp.asarray(dists).flatten()
        a_global = db_subset_indices[cp.asarray(neighbors).flatten()]
        b_global = cp.repeat(current_q_indices, k_local)
        
        # Filter self-matches
        if exclude_self:
            mask = dists > 1e-4
            dists, a_global, b_global = dists[mask], a_global[mask], b_global[mask]
        
        # Merge with running top-K
        all_d = cp.concatenate([best_dists, dists])
        all_a = cp.concatenate([best_a, a_global])
        all_b = cp.concatenate([best_b, b_global])
        
        top_num = min(k_global * 2, all_d.shape[0])
        top_idx = cp.argpartition(all_d, top_num)[:top_num]
        sort_idx_val = cp.argsort(all_d[top_idx])
        final_idx = top_idx[sort_idx_val]
        
        best_dists = all_d[final_idx]
        best_a = all_a[final_idx]
        best_b = all_b[final_idx]
    
    # Deduplicate results
    unique_results = {}
    for d, a, b in zip(best_dists.tolist(), best_a.tolist(), best_b.tolist()):
        pair_key = (int(b), int(a))
        if pair_key not in unique_results:
            unique_results[pair_key] = d
    
    results = sorted(unique_results.items(), key=lambda x: x[1])[:k_global]
    results = [(b, a, d) for (b, a), d in results]
    
    cp.cuda.Stream.null.synchronize()
    return results, time.perf_counter() - t_start


def ivf_flat_search(
    vectors_A: cp.ndarray,
    vectors_B: cp.ndarray,
    k_global: int = 10,
    n_lists: int = 4096,
    n_probes: int = 64,
    exclude_self: bool = True,
    batch_size: int = 50_000,
) -> tuple[list, float]:
    """
    IVF-Flat search using cuVS with batching to avoid OOM.
    
    Returns:
        results: List of (b_idx, a_idx, distance) tuples
        elapsed: Total time in seconds
    """
    t_start = time.perf_counter()
    M = vectors_B.shape[0]
    
    # Build index
    build_params = ivf_flat.IndexParams(n_lists=n_lists)
    index = ivf_flat.build(build_params, vectors_A)
    search_params = ivf_flat.SearchParams(n_probes=n_probes)
    
    k_per_query = min(k_global + 5, 100)  # Enough to filter self-matches
    
    # Running top-K state
    best_dists = cp.full(k_global, cp.inf, dtype=cp.float32)
    best_a = cp.zeros(k_global, dtype=cp.int64)
    best_b = cp.zeros(k_global, dtype=cp.int64)
    
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        batch_B = vectors_B[b_start:b_end]
        
        distances, neighbors = ivf_flat.search(search_params, index, batch_B, k=k_per_query)
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)
        
        if exclude_self:
            distances[distances <= 1e-4] = cp.inf
        
        # Flatten and create global B indices
        flat_distances = distances.flatten()
        flat_neighbors = neighbors.flatten()
        flat_b_indices = cp.repeat(cp.arange(b_start, b_end, dtype=cp.int64), k_per_query)
        
        # Merge with running top-K
        all_d = cp.concatenate([best_dists, flat_distances])
        all_a = cp.concatenate([best_a, flat_neighbors])
        all_b = cp.concatenate([best_b, flat_b_indices])
        
        # Select new top-K
        if all_d.shape[0] > k_global:
            top_idx = cp.argpartition(all_d, k_global)[:k_global]
            top_idx = top_idx[cp.argsort(all_d[top_idx])]
        else:
            top_idx = cp.argsort(all_d)
        
        best_dists = all_d[top_idx]
        best_a = all_a[top_idx]
        best_b = all_b[top_idx]
        
        del distances, neighbors, flat_distances
        cp.get_default_memory_pool().free_all_blocks()
    
    # Extract final results
    results = []
    for i in range(len(best_dists)):
        b_idx = int(best_b[i])
        a_idx = int(best_a[i])
        dist = float(best_dists[i])
        if dist < cp.inf:
            results.append((b_idx, a_idx, dist))
    
    cp.cuda.Stream.null.synchronize()
    return results, time.perf_counter() - t_start


def brute_force_search(
    vectors_A: cp.ndarray,
    vectors_B: cp.ndarray,
    k_global: int = 10,
    exclude_self: bool = True,
    batch_size: int = 50_000,
) -> tuple[list, float]:
    """
    Brute force search (ground truth) with batching to avoid OOM.
    
    Returns:
        results: List of (b_idx, a_idx, distance) tuples
        elapsed: Total time in seconds
    """
    t_start = time.perf_counter()
    M = vectors_B.shape[0]
    
    # Need extra neighbors per query to filter self-matches
    k_per_query = min(k_global + 5, 100)  # Enough to find non-self matches
    
    index = brute_force.build(vectors_A)
    
    # Running top-K state
    best_dists = cp.full(k_global, cp.inf, dtype=cp.float32)
    best_a = cp.zeros(k_global, dtype=cp.int64)
    best_b = cp.zeros(k_global, dtype=cp.int64)
    
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        batch_B = vectors_B[b_start:b_end]
        batch_size_actual = b_end - b_start
        
        distances, neighbors = brute_force.search(index, batch_B, k=k_per_query)
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)
        
        if exclude_self:
            distances[distances <= 1e-4] = cp.inf
        
        # Flatten and create global B indices
        flat_distances = distances.flatten()
        flat_neighbors = neighbors.flatten()
        flat_b_indices = cp.repeat(cp.arange(b_start, b_end, dtype=cp.int64), k_per_query)
        
        # Merge with running top-K
        all_d = cp.concatenate([best_dists, flat_distances])
        all_a = cp.concatenate([best_a, flat_neighbors])
        all_b = cp.concatenate([best_b, flat_b_indices])
        
        # Select new top-K
        if all_d.shape[0] > k_global:
            top_idx = cp.argpartition(all_d, k_global)[:k_global]
            top_idx = top_idx[cp.argsort(all_d[top_idx])]
        else:
            top_idx = cp.argsort(all_d)
        
        best_dists = all_d[top_idx]
        best_a = all_a[top_idx]
        best_b = all_b[top_idx]
        
        del distances, neighbors, flat_distances
        cp.get_default_memory_pool().free_all_blocks()
    
    # Extract final results
    results = []
    for i in range(len(best_dists)):
        b_idx = int(best_b[i])
        a_idx = int(best_a[i])
        dist = float(best_dists[i])
        if dist < cp.inf:
            results.append((b_idx, a_idx, dist))
    
    cp.cuda.Stream.null.synchronize()
    return results, time.perf_counter() - t_start


def print_results(results: list, method: str, elapsed: float, k: int, show_pairs: int = 10):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"{method}")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Found {len(results):,} pairs")
    print(f"\nTop {show_pairs} Pairs:")
    for b, a, d in results[:show_pairs]:
        print(f"  B[{b:<7}] <-> A[{a:<7}] (Distance: {d:.5f})")


def compute_recall(results: list, ground_truth: list) -> float:
    """Compute recall of results against ground truth."""
    result_pairs = set((b, a) for b, a, _ in results)
    gt_pairs = set((b, a) for b, a, _ in ground_truth)
    
    if len(gt_pairs) == 0:
        return 0.0
    
    intersection = result_pairs & gt_pairs
    return len(intersection) / len(gt_pairs)


# ============== Main ==============
if __name__ == "__main__":
    print("Loading SIFT1M Base Set...")
    vectors = read_fvecs('/home/william/thesis_ws/datasets/sift/sift_base.fvecs')
    
    # Self-join setup
    vectors_A = vectors
    vectors_B = vectors
    k_global = 10_000
    
    print(f"\nSelf-Join: {vectors_A.shape[0]:,} x {vectors_B.shape[0]:,} vectors")
    print(f"Finding top {k_global:,} closest pairs\n")
    
    # Run brute force first as ground truth
    print("Running Brute Force (ground truth)...")
    results_brute, time_brute = brute_force_search(vectors_A, vectors_B, k_global)
    print_results(results_brute, "Brute Force Search (Ground Truth)", time_brute, k_global)
    
    # Run IVF-Flat
    print("\nRunning IVF-Flat...")
    results_ivf, time_ivf = ivf_flat_search(vectors_A, vectors_B, k_global)
    recall_ivf = compute_recall(results_ivf, results_brute)
    print_results(results_ivf, "IVF-Flat Search", time_ivf, k_global)
    print(f"Recall vs Ground Truth: {recall_ivf:.4f} ({recall_ivf*100:.2f}%)")
    
    # Run Centroid Search
    print("\nRunning Centroid Search...")
    results_centroid, time_centroid = centroid_search(vectors_A, vectors_B, k_global)
    recall_centroid = compute_recall(results_centroid, results_brute)
    print_results(results_centroid, "Centroid-Restricted Search", time_centroid, k_global)
    print(f"Recall vs Ground Truth: {recall_centroid:.4f} ({recall_centroid*100:.2f}%)")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Time (s)':<12} {'Recall':<12}")
    print(f"{'-'*54}")
    print(f"{'Brute Force (GT)':<30} {time_brute:<12.4f} {'1.0000':<12}")
    print(f"{'IVF-Flat Search':<30} {time_ivf:<12.4f} {recall_ivf:<12.4f}")
    print(f"{'Centroid Search':<30} {time_centroid:<12.4f} {recall_centroid:<12.4f}")

