"""
Centroid-aware batching strategy:
  1. Cluster the dataset with KMeans (on a training subsample)
  2. Assign every point to a cluster with balanced capacity caps
  3. Prune chunk pairs whose centroid distance > threshold
  4. Dispatch surviving pairs to the join algorithm

The schedule step (_build_schedule) is deliberately separated so a smarter
scheduler (priority ordering, concurrent dispatch, etc.) can be swapped in
later without touching clustering or execution.
"""
import math
import os
import time

import cupy as cp
import numpy as np
from cupyx.profiler import time_range
from cuvs.cluster import kmeans
from cuvs.neighbors import brute_force


# ── 1. Cluster & balanced assignment ─────────────────────────────

def _cluster_and_assign(dataset, n_clusters, sample_fraction=0.1):
    """KMeans on a subsample, then greedy balanced assignment.

    Each centroid bucket is capped at ceil(N / n_clusters) points.
    Overflow goes to the next-closest non-full centroid.

    Args:
        dataset: numpy float32 array (N, D).
        n_clusters: number of clusters / chunks.
        sample_fraction: fraction of dataset to subsample for KMeans.

    Returns:
        chunks: list[np.ndarray] — chunks[i] = global indices for cluster i.
        centroids: np.ndarray (n_clusters, D) — centroid vectors on CPU.
    """
    N, D = dataset.shape
    n_clusters = min(n_clusters, N)
    capacity = math.ceil(N / n_clusters)

    # ── KMeans on subsample ──
    with time_range("centroid_batch/kmeans_fit", color_id=1):
        n_samples = max(n_clusters, int(N * sample_fraction))
        if n_samples < N:
            sample_idx = np.random.choice(N, size=n_samples, replace=False)
            sample = cp.asarray(dataset[sample_idx])
        else:
            sample = cp.asarray(dataset)

        params = kmeans.KMeansParams(n_clusters=n_clusters)
        cp.get_default_memory_pool().free_all_blocks()
        centroids, _, _ = kmeans.fit(params, sample)
        cp.cuda.Stream.null.synchronize()

    # ── Rank every point by distance to nearest centroids (batched) ──
    with time_range("centroid_batch/assign_points", color_id=2):
        c_index = brute_force.build(centroids)

        # Use a small k to avoid huge output allocations.
        # Most points settle into top-1; overflow rarely needs more.
        rank_k = min(10, n_clusters)
        assign_batch = 1_000_000  # process this many points at a time

        all_ranked = np.empty((N, rank_k), dtype=np.int32)
        all_first_dist = np.empty(N, dtype=np.float32)

        for start in range(0, N, assign_batch):
            end = min(start + assign_batch, N)
            cp.get_default_memory_pool().free_all_blocks()
            
            gpu_chunk = cp.asarray(dataset[start:end])
            dists_chunk, idx_chunk = brute_force.search(
                c_index, gpu_chunk, k=rank_k
            )
            all_ranked[start:end] = cp.asarray(idx_chunk).get()
            all_first_dist[start:end] = cp.asarray(dists_chunk)[:, 0].get()
            
            del gpu_chunk

        cp.cuda.Stream.null.synchronize()

        # ── Greedy balanced assignment ──
        # Sort points by distance to 1st-choice centroid
        # so "confident" points get placed first
        point_order = np.argsort(all_first_dist)

        assignments = np.full(N, -1, dtype=np.int32)
        counts = np.zeros(n_clusters, dtype=np.int32)

        for pt in point_order:
            for rank in range(rank_k):
                c = all_ranked[pt, rank]
                if counts[c] < capacity:
                    assignments[pt] = c
                    counts[c] += 1
                    break
            else:
                # All top-k centroids full — assign to any non-full centroid
                for c in range(n_clusters):
                    if counts[c] < capacity:
                        assignments[pt] = c
                        counts[c] += 1
                        break

        # Build chunk index lists
        chunks = [np.where(assignments == c)[0] for c in range(n_clusters)]
        centroids_cpu = cp.asnumpy(cp.asarray(centroids))

    return chunks, centroids_cpu


# ── 2. Build schedule (scheduler hook) ───────────────────────────

def _build_schedule(centroids, threshold, self_join):
    """Decide which chunk pairs to compare based on centroid proximity.

    Args:
        centroids: np.ndarray (K, D) — centroid vectors.
        threshold: squared L2 distance threshold.
        self_join: if True, only upper-triangle pairs (j >= i).

    Returns:
        schedule: list of (i, j) tuples to compare.
        centroid_dists: (K, K) pairwise centroid distance matrix (for stats).
    """
    with time_range("centroid_batch/build_schedule", color_id=3):
        K = centroids.shape[0]
        gpu_c = cp.asarray(centroids)

        # Pairwise squared L2 between centroids
        c_norms = cp.sum(gpu_c ** 2, axis=1)
        centroid_dists = c_norms[:, None] - 2.0 * (gpu_c @ gpu_c.T) + c_norms[None, :]
        centroid_dists = cp.maximum(centroid_dists, 0.0)
        centroid_dists_cpu = cp.asnumpy(centroid_dists)

        schedule = []
        for i in range(K):
            start_j = i if self_join else 0
            for j in range(start_j, K):
                if centroid_dists_cpu[i, j] <= threshold:
                    schedule.append((i, j))

    return schedule, centroid_dists_cpu


# ── 3. Execute schedule ──────────────────────────────────────────

def _execute_schedule(schedule, chunks, dataset, join_algorithm,
                      threshold, self_join, result_dir, params):
    """Run the join algorithm on each scheduled chunk pair."""
    a_path = os.path.join(result_dir, "a_indices.bin")
    b_path = os.path.join(result_dir, "b_indices.bin")
    d_path = os.path.join(result_dir, "distances.bin")

    total_pairs = 0
    n_tiles = 0
    total_pairs_compared = 0
    total_build_time = 0.0

    with open(a_path, "wb") as fa, \
         open(b_path, "wb") as fb, \
         open(d_path, "wb") as fd:

        for idx, (i, j) in enumerate(schedule):
            chunk_A_indices = chunks[i]
            chunk_B_indices = chunks[j]

            if len(chunk_A_indices) == 0 or len(chunk_B_indices) == 0:
                continue

            chunk_A = dataset[chunk_A_indices]
            chunk_B = dataset[chunk_B_indices]

            is_diagonal = self_join and (i == j)

            with time_range(f"centroid_batch/tile_{n_tiles}", color_id=6):
                result = join_algorithm(
                    chunk_A, chunk_B, threshold,
                    self_join_diagonal=is_diagonal,
                    params=params,
                )

            if len(result) == 4:
                a_idx, b_idx, dists, extra_stats = result
            else:
                a_idx, b_idx, dists = result
                extra_stats = {}

            if "build_time_s" in extra_stats:
                total_build_time += extra_stats["build_time_s"]

            if "pairs_compared" in extra_stats:
                total_pairs_compared += extra_stats["pairs_compared"]
            else:
                total_pairs_compared += len(chunk_A) * len(chunk_B)

            n_tiles += 1

            if len(a_idx) > 0:
                # Remap local indices → global indices
                global_a = chunk_A_indices[a_idx].astype(np.int64)
                global_b = chunk_B_indices[b_idx].astype(np.int64)

                fa.write(global_a.tobytes())
                fb.write(global_b.tobytes())
                fd.write(dists.astype(np.float32).tobytes())
                total_pairs += len(a_idx)

            if (idx + 1) % 50 == 0 or idx == len(schedule) - 1:
                print(f"  tile {idx+1}/{len(schedule)} done, "
                      f"{total_pairs:,} pairs so far")

    return total_pairs, n_tiles, total_pairs_compared, total_build_time


# ── Top-level entry point ────────────────────────────────────────

def centroid_batch(dataset_A, dataset_B, join_algorithm, self_join,
                   output_dir, method_name, params):
    """Centroid-aware batching with distance-based pair pruning.

    All parameters are read from params:
      - params.batch_sizes[method_name] → chunk size (n_clusters = ceil(N / batch_size))
      - params.centroid_batch.centroid_threshold → pruning threshold
      - params.centroid_batch.sample_fraction → KMeans subsample ratio
      - params.threshold → join distance threshold
    """
    threshold = params.threshold
    batch_size = params.batch_sizes[method_name]
    centroid_threshold = params.centroid_batch.centroid_threshold
    sample_fraction = params.centroid_batch.sample_fraction

    N = dataset_A.shape[0]
    n_clusters = max(1, math.ceil(N / batch_size))

    result_dir = os.path.join(output_dir, method_name)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{method_name}: N={N:,}, D={dataset_A.shape[1]}, "
          f"threshold={threshold}, centroid_threshold={centroid_threshold}, "
          f"batch_size={batch_size:,}, n_clusters={n_clusters}")

    with time_range("centroid_batch", color_id=0):
        # 1. Cluster and assign
        t0 = time.time()
        chunks, centroids = _cluster_and_assign(dataset_A, n_clusters,
                                                sample_fraction=sample_fraction)
        cp.cuda.Stream.null.synchronize()
        cluster_time = time.time() - t0

        chunk_sizes = [len(c) for c in chunks]
        print(f"  clustering: {cluster_time:.2f}s, "
              f"chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, "
              f"mean={np.mean(chunk_sizes):.0f}")

        # 2. Build schedule
        schedule, centroid_dists = _build_schedule(
            centroids, centroid_threshold, self_join
        )

        total_possible = n_clusters * (n_clusters + 1) // 2 if self_join \
                         else n_clusters * n_clusters
        print(f"  schedule: {len(schedule)}/{total_possible} pairs "
              f"({100 * len(schedule) / max(total_possible, 1):.1f}% of all)")

        # 3. Execute schedule
        total_pairs, n_tiles, total_pairs_compared, total_build_time = \
            _execute_schedule(
                schedule, chunks, dataset_A, join_algorithm,
                threshold, self_join, result_dir, params
            )

    print(f"  -> saved to {result_dir}")

    batch_metadata = {
        "cluster_time_s": cluster_time,
        "n_clusters": n_clusters,
        "scheduled_tiles": len(schedule),
        "total_possible_tiles": total_possible,
        "prune_ratio": 1.0 - len(schedule) / max(total_possible, 1),
    }
    if total_build_time > 0:
        batch_metadata["build_time_s"] = total_build_time

    return result_dir, total_pairs, n_tiles, total_pairs_compared, batch_metadata
