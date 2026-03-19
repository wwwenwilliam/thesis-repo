"""
Simplest batching strategy:
- Splits dataset into N chunks of batch_size
- Iterates over all tile pairs (i, j), for self-join: j >= i only
- Calls join_algorithm on each pair
- Writes results to disk as they go (binary files)
"""
import os

import numpy as np
from cupyx.profiler import time_range


def simple_batch(dataset_A, dataset_B, join_algorithm, threshold,
                 batch_size, self_join, output_dir, method_name):
    """Run a join algorithm over tiled batch pairs, writing results to disk.

    Pure batching — no timing or metadata. The caller (benchmark.py) is
    responsible for instrumentation.

    Args:
        dataset_A: numpy float32 array, shape (N, d).
        dataset_B: numpy float32 array, shape (M, d).
        join_algorithm: Callable(chunk_A, chunk_B, threshold, self_join_diagonal) -> (a_idx, b_idx, dists).
        threshold: Distance threshold to pass to join_algorithm.
        batch_size: Number of vectors per tile.
        self_join: If True, only compute upper-triangle tiles (j >= i).
        output_dir: Root output directory.
        method_name: Subdirectory name for this method's results.

    Returns:
        (result_dir, total_pairs, n_tiles, total_pairs_compared)
    """
    N = dataset_A.shape[0]
    M = dataset_B.shape[0]

    result_dir = os.path.join(output_dir, method_name)
    os.makedirs(result_dir, exist_ok=True)

    a_path = os.path.join(result_dir, "a_indices.bin")
    b_path = os.path.join(result_dir, "b_indices.bin")
    d_path = os.path.join(result_dir, "distances.bin")

    total_pairs = 0
    n_tiles = 0
    total_pairs_compared = 0
    total_build_time = 0.0

    print(f"\n{method_name}: N={N:,}, M={M:,}, D={dataset_A.shape[1]}, "
          f"threshold={threshold}, batch_size={batch_size:,}")

    with open(a_path, "wb") as fa, open(b_path, "wb") as fb, open(d_path, "wb") as fd:
        for b_start in range(0, M, batch_size):
            b_end = min(b_start + batch_size, M)
            chunk_B = dataset_B[b_start:b_end]

            start_a = b_start if self_join else 0

            for a_start in range(start_a, N, batch_size):
                a_end = min(a_start + batch_size, N)
                chunk_A = dataset_A[a_start:a_end]

                is_diagonal = self_join and (a_start == b_start)

                with time_range(f"batch/tile_{n_tiles}", color_id=6):
                    result = join_algorithm(
                        chunk_A, chunk_B, threshold,
                        self_join_diagonal=is_diagonal
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
                    total_pairs_compared += (a_end - a_start) * (b_end - b_start)

                n_tiles += 1

                if len(a_idx) > 0:
                    global_a = (a_idx + a_start).astype(np.int64)
                    global_b = (b_idx + b_start).astype(np.int64)

                    fa.write(global_a.tobytes())
                    fb.write(global_b.tobytes())
                    fd.write(dists.astype(np.float32).tobytes())
                    total_pairs += len(a_idx)

            print(f"  batch b=[{b_start:,}:{b_end:,}] done, "
                  f"{total_pairs:,} pairs so far")

    print(f"  -> saved to {result_dir}")
    
    batch_metadata = {}
    if total_build_time > 0:
         batch_metadata["build_time_s"] = total_build_time

    return result_dir, total_pairs, n_tiles, total_pairs_compared, batch_metadata
