"""
Post-processing for similarity join results.

Reads result directories, computes statistics, compares methods
against brute force ground truth, and writes a summary file.
"""
import json
import os

import numpy as np

DTYPE_IDX = np.int64
DTYPE_DIST = np.float32


def _load_metadata(result_dir):
    """Load metadata.json from a result directory."""
    path = os.path.join(result_dir, "metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def _mmap_results(result_dir, pair_count):
    """Memory-map the result binary files."""
    if pair_count == 0:
        return (np.empty(0, dtype=DTYPE_IDX),
                np.empty(0, dtype=DTYPE_IDX),
                np.empty(0, dtype=DTYPE_DIST))

    a = np.memmap(os.path.join(result_dir, "a_indices.bin"),
                  dtype=DTYPE_IDX, mode='r', shape=(pair_count,))
    b = np.memmap(os.path.join(result_dir, "b_indices.bin"),
                  dtype=DTYPE_IDX, mode='r', shape=(pair_count,))
    d = np.memmap(os.path.join(result_dir, "distances.bin"),
                  dtype=DTYPE_DIST, mode='r', shape=(pair_count,))
    return a, b, d


def _load_packed_gt(result_dir, pair_count, chunk_size=10_000_000):
    """Load ground truth pairs into a sorted numpy array of unique packed uint64s."""
    a, b, _ = _mmap_results(result_dir, pair_count)
    packed = np.empty(pair_count, dtype=np.uint64)
    for start in range(0, pair_count, chunk_size):
        end = min(start + chunk_size, pair_count)
        packed[start:end] = (a[start:end].astype(np.uint64) << 32) | b[start:end].astype(np.uint64)
    return np.unique(packed)


def _compute_stats(result_dir, metadata):
    """Compute distance statistics for a single method."""
    pair_count = metadata.get("pair_count", 0)
    stats = {
        "method": metadata.get("method", "unknown"),
        "pair_count": pair_count,
        "time_s": metadata.get("time_s", 0),
        "threshold": metadata.get("threshold", 0),
        "batch_size": metadata.get("batch_size", 0),
        "n_tiles": metadata.get("n_tiles", 0),
        "total_pairs_compared": metadata.get("total_pairs_compared", 0),
    }

    if "build_time_s" in metadata:
        stats["build_time_s"] = metadata["build_time_s"]

    if pair_count > 0:
        a_idx, b_idx, dists = _mmap_results(result_dir, pair_count)
        
        # Determine number of vectors to use for bincount length
        N = metadata.get("N", 0)
        M = metadata.get("M", 0)
        self_join = metadata.get("self_join", False)

        if self_join and N > 0:
            # Self-join: upper triangle (a > b) is stored, so both counts matter
            # We bincount everything to get the full degree of each point
            degrees_a = np.bincount(a_idx, minlength=N)
            degrees_b = np.bincount(b_idx, minlength=N)
            neighbors_per_point = degrees_a + degrees_b
        elif M > 0:
            # Cross-join: we want neighbors per query point (in B)
            neighbors_per_point = np.bincount(b_idx, minlength=M)
        else:
            # Fallback if metadata is missing sizes
            neighbors_per_point = np.bincount(b_idx)
            
        stats["n_neighbors_min"] = float(np.min(neighbors_per_point))
        stats["n_neighbors_max"] = float(np.max(neighbors_per_point))
        stats["n_neighbors_mean"] = float(np.mean(neighbors_per_point))
        stats["n_neighbors_std"] = float(np.std(neighbors_per_point))

    return stats


def process_results(output_dir, ground_truth_method="brute_force"):
    """Read all method results, compute stats, compare recall, and print summary.

    Args:
        output_dir: Root results directory containing method subdirectories.
        ground_truth_method: Method name to use as recall reference.

    Returns:
        List of per-method stat dicts.
    """
    methods = sorted(
        name for name in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, name))
        and os.path.exists(os.path.join(output_dir, name, "metadata.json"))
    )

    if not methods:
        print(f"No results found in {output_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"Results Summary — {output_dir}")
    print(f"{'='*60}")

    all_stats = []
    gt_dir = os.path.join(output_dir, ground_truth_method)
    gt_packed = None

    if ground_truth_method in methods:
        gt_meta = _load_metadata(gt_dir)
        gt_count = gt_meta.get("pair_count", 0)
        if gt_count > 0:
            gt_packed = _load_packed_gt(gt_dir, gt_count)

    for method in methods:
        result_dir = os.path.join(output_dir, method)
        metadata = _load_metadata(result_dir)
        stats = _compute_stats(result_dir, metadata)

        if (gt_packed is not None and method != ground_truth_method
                and stats["pair_count"] > 0):
            
            # Batch method results against ground truth array to limit RAM
            gt_matched = np.zeros(len(gt_packed), dtype=bool)
            a, b, _ = _mmap_results(result_dir, stats["pair_count"])
            chunk_size = 10_000_000
            
            for start in range(0, stats["pair_count"], chunk_size):
                end = min(start + chunk_size, stats["pair_count"])
                chunk_packed = (a[start:end].astype(np.uint64) << 32) | b[start:end].astype(np.uint64)
                
                # Check intersection using fast binary search
                idx = np.searchsorted(gt_packed, chunk_packed)
                valid = idx < len(gt_packed)
                valid_indices = idx[valid]
                matches = chunk_packed[valid] == gt_packed[valid_indices]
                gt_matched[valid_indices[matches]] = True

            stats["recall"] = float(gt_matched.sum() / len(gt_packed))

        all_stats.append(stats)

        print(f"\n{method}:")
        print(f"  Time:  {stats['time_s']:.4f}s")
        if "build_time_s" in stats:
            print(f"  Index Build Time: {stats['build_time_s']:.4f}s")
        print(f"  Pairs: {stats['pair_count']:,}")
        if stats["pair_count"] > 0 and "n_neighbors_mean" in stats:
            print(f"  Neighbours/pt: min={stats['n_neighbors_min']:.2f}  "
                  f"max={stats['n_neighbors_max']:.2f}  "
                  f"mean={stats['n_neighbors_mean']:.2f}  "
                  f"std={stats['n_neighbors_std']:.2f}")
        if "recall" in stats:
            print(f"  Recall vs {ground_truth_method}: {stats['recall']:.4f}")

    return all_stats


def dump_statistics(output_dir, stats_list=None,
                    ground_truth_method="brute_force"):
    """Write statistics.json to the output directory.

    If stats_list is not provided, process_results is called first.
    """
    if stats_list is None:
        stats_list = process_results(output_dir, ground_truth_method)

    out_path = os.path.join(output_dir, "statistics.json")
    with open(out_path, "w") as f:
        json.dump(stats_list, f, indent=2)
    print(f"\n-> Statistics written to {out_path}")
