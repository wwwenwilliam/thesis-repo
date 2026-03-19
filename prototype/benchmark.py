"""
Vector similarity join benchmark runner.

Flow:
  1. Read dataset(s) into numpy array
  2. Run join method(s) via batching strategy (results written to disk)
  3. Process results and dump statistics
"""
import json
import sys
import os
import time

import numpy as np
from cupyx.profiler import time_range

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.read_dataset import read_u8bin, read_fvecs
from batching.simple_batch import simple_batch
from joins.brute_force import brute_force_join
from joins.cuvs_knn import cuvs_cagra_join, cuvs_ivf_pq_join, cuvs_ivf_flat_join, cuvs_brute_force_join
from joins.centroid_join import centroid_join
from util.process_results import process_results, dump_statistics

# ── Configuration ───────────────────────────────────────────────
DATASET_A_PATH = "/home/william/thesis_ws/datasets/sift/learn.100M.u8bin"
DATASET_A_FMT  = "u8bin"
NUM_VECTORS_A  = 1_000_000

DATASET_B_PATH = ""            # leave empty for self-join
DATASET_B_FMT  = "u8bin"
NUM_VECTORS_B  = 100_000

OUTPUT_DIR   = "./results/1M"
THRESHOLD    = 10_000.0
METHODS      = ["centroid_join", "cuvs_ivf_flat"]

ALGORITHMS = {
    "brute_force": (brute_force_join, simple_batch, 25_000),
    "cuvs_ivf_flat": (cuvs_ivf_flat_join, simple_batch, 100_000),
    "cuvs_cagra": (cuvs_cagra_join, simple_batch, 20_000),
    "cuvs_brute_force": (cuvs_brute_force_join, simple_batch, 100_000),
    "centroid_join": (centroid_join, simple_batch, 1_000_000),
}


def load_dataset(path, fmt, max_vectors=None):
    """Load a dataset based on format string."""
    loaders = {
        "u8bin": read_u8bin,
        "fvecs": read_fvecs,
    }
    if fmt not in loaders:
        raise ValueError(f"Unknown format '{fmt}', expected one of {list(loaders)}")
    return loaders[fmt](path, max_vectors=max_vectors)


def main():
    print("Loading dataset A...")
    dataset_A = load_dataset(DATASET_A_PATH, DATASET_A_FMT, max_vectors=NUM_VECTORS_A)
    print(f"Dataset A: {dataset_A.shape[0]:,} x {dataset_A.shape[1]}D")

    self_join = not DATASET_B_PATH
    if self_join:
        print("Self-join mode (no dataset B specified)\n")
        dataset_B = dataset_A
    else:
        print(f"\nLoading dataset B...")
        dataset_B = load_dataset(DATASET_B_PATH, DATASET_B_FMT, max_vectors=NUM_VECTORS_B)
        print(f"Dataset B: {dataset_B.shape[0]:,} x {dataset_B.shape[1]}D\n")

    rng = np.random.default_rng(42)
    dataset_A = dataset_A[rng.permutation(len(dataset_A))]
    if not self_join:
        dataset_B = dataset_B[rng.permutation(len(dataset_B))]
    else:
        dataset_B = dataset_A
    print("Shuffled vectors (seed=42)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method_name in METHODS:
        if method_name not in ALGORITHMS:
            print(f"Skipping unknown method: {method_name}")
            continue

        join_fn, batch_fn, batch_size = ALGORITHMS[method_name]

        t0 = time.perf_counter()
        with time_range(f"benchmark/{method_name}", color_id=7):
            result_dir, total_pairs, n_tiles, pairs_compared, batch_metadata = batch_fn(
                dataset_A=dataset_A,
                dataset_B=dataset_B,
                join_algorithm=join_fn,
                threshold=THRESHOLD,
                batch_size=batch_size,
                self_join=self_join,
                output_dir=OUTPUT_DIR,
                method_name=method_name,
            )
        elapsed = time.perf_counter() - t0

        metadata = {
            "method": method_name,
            "pair_count": total_pairs,
            "time_s": elapsed,
            "threshold": THRESHOLD,
            "batch_size": batch_size,
            "n_tiles": n_tiles,
            "total_pairs_compared": pairs_compared,
            "self_join": self_join,
            "N": dataset_A.shape[0],
            "M": dataset_B.shape[0],
            "D": int(dataset_A.shape[1]),
        }
        
        # Merge any extra batch statistics (e.g. cuvs index build_time_s)
        metadata.update(batch_metadata)

        with open(os.path.join(result_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{method_name} finished: {elapsed:.4f}s, "
              f"{n_tiles} tiles, {total_pairs:,} pairs found")
        print(f"  pairs compared: {pairs_compared:,} "
              f"({pairs_compared / 1e9:.2f}B)")

    stats = process_results(OUTPUT_DIR)
    dump_statistics(OUTPUT_DIR, stats)


if __name__ == "__main__":
    main()
