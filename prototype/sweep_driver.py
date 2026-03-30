"""
Parameter sweep driver.

Runs brute_force once for ground truth, then sweeps cuVS and centroid_join
parameters, computing recall and timing for each configuration.

Results are stored in the same format as benchmark.py — each sweep point
gets its own subdirectory with a_indices.bin, b_indices.bin, distances.bin,
and metadata.json.

Usage:
    python sweep_driver.py
"""
import csv
import dataclasses
import json
import os
import sys
import time
from itertools import product

import numpy as np
from cupyx.profiler import time_range

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import Params
from util.read_dataset import read_u8bin, read_fvecs, read_fbin, read_bvecs
from batching.simple_batch import simple_batch

from joins.brute_force import brute_force_join
from joins.cuvs_knn import cuvs_cagra_join, cuvs_ivf_flat_join
from joins.centroid_join import centroid_join
from util.process_results import process_results, dump_statistics

# ── Dataset configuration ───────────────────────────────────────
DATASET_A_PATH = "/home/william/thesis_ws/thesis-repo/datasets/sift/bigann_learn.bvecs"
DATASET_A_FMT  = "bvecs"
NUM_VECTORS_A  = 1_000_000

DATASET_B_PATH = ""            # empty = self-join
DATASET_B_FMT  = "bvecs"
NUM_VECTORS_B  = 1_000_000

OUTPUT_DIR = "./results/SIFT/SWEEP_1024"

# ── Base parameters (non-swept values) ──────────────────────────
BASE_PARAMS = Params(
    threshold=1024.0,
)

# ── Sweep ranges (edit these independently per method) ──────────
CUVS_CAGRA_K_SWEEP = [64]
CUVS_CAGRA_ITOPK_SWEEP = [2, 4, 8, 16, 32, 64]

CUVS_IVF_FLAT_K_SWEEP = [64, 128, 256]
CUVS_IVF_FLAT_NLISTS_SWEEP = [128, 256, 512, 1024]
CUVS_IVF_FLAT_NPROBES_SWEEP = [2, 4, 8, 16, 20, 32]

CENTROID_N_CLUSTERS_SWEEP = [128, 256, 512, 1024, 2048]

CENTROID_K_DB_CANDIDATES_SWEEP = [1024, 2048, 4096, 8192, 16384, 32768]


# ── Algorithm mapping ───────────────────────────────────────────
JOIN_FNS = {
    "cuvs_cagra":    cuvs_cagra_join,
    "cuvs_ivf_flat": cuvs_ivf_flat_join,
    "centroid_join":  centroid_join,
}


def load_dataset(path, fmt, max_vectors=None):
    """Load a dataset based on format string."""
    loaders = {
        "u8bin": read_u8bin,
        "fvecs": read_fvecs,
        "fbin": read_fbin,
        "bvecs": read_bvecs,
    }
    if fmt not in loaders:
        raise ValueError(f"Unknown format '{fmt}', expected one of {list(loaders)}")
    return loaders[fmt](path, max_vectors=max_vectors)


def _sweep_point_exists(output_dir, method_name):
    """Check if a sweep point already has results (for resume support)."""
    meta_path = os.path.join(output_dir, method_name, "metadata.json")
    return os.path.exists(meta_path)


def _run_single(dataset_A, dataset_B, self_join, output_dir,
                method_name, base_method, params):
    """Run a single sweep point: execute the join and write metadata."""
    join_fn = JOIN_FNS.get(base_method)
    if join_fn is None and base_method == "brute_force":
        join_fn = brute_force_join

    # Ensure batch_sizes has an entry for this sweep method name
    params.batch_sizes[method_name] = params.batch_sizes[base_method]

    print(f"\n{'='*60}")
    print(f"  SWEEP POINT: {method_name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    with time_range(f"sweep/{method_name}", color_id=7):
        result_dir, total_pairs, n_tiles, pairs_compared, batch_metadata = simple_batch(
            dataset_A=dataset_A,
            dataset_B=dataset_B,
            join_algorithm=join_fn,
            self_join=self_join,
            output_dir=output_dir,
            method_name=method_name,
            params=params,
        )
    elapsed = time.perf_counter() - t0

    metadata = {
        "method": method_name,
        "base_method": base_method,
        "pair_count": total_pairs,
        "time_s": elapsed,
        "n_tiles": n_tiles,
        "total_pairs_compared": pairs_compared,
        "self_join": self_join,
        "N": dataset_A.shape[0],
        "M": dataset_B.shape[0],
        "D": int(dataset_A.shape[1]),
        "params": params.to_dict(),
    }
    metadata.update(batch_metadata)

    with open(os.path.join(result_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  {method_name} finished: {elapsed:.4f}s, "
          f"{n_tiles} tiles, {total_pairs:,} pairs found")
    return metadata


def _make_params(**overrides):
    """Create a fresh Params from BASE_PARAMS with field-level overrides.

    Supports nested override keys like 'cuvs_cagra' by passing the
    sub-dataclass via dataclasses.replace.
    """
    return dataclasses.replace(BASE_PARAMS, **overrides)


def generate_sweep_summary(output_dir):
    """Parse all metadata.json files and write a tidy sweep_summary.csv."""
    rows = []
    for name in sorted(os.listdir(output_dir)):
        meta_path = os.path.join(output_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        row = {
            "method_name": name,
            "base_method": meta.get("base_method", name),
            "time_s": meta.get("time_s", 0),
            "pair_count": meta.get("pair_count", 0),
        }

        # Extract swept parameters from the stored params
        stored_params = meta.get("params", {})
        base = row["base_method"]
        if base in ("cuvs_cagra", "cuvs_ivf_flat"):
            row["k"] = stored_params.get(base, {}).get("k", "")
            if base == "cuvs_cagra":
                row["itopk_size"] = stored_params.get(base, {}).get("itopk_size", "")
            elif base == "cuvs_ivf_flat":
                row["n_lists"] = stored_params.get(base, {}).get("n_lists", "")
                row["n_probes"] = stored_params.get(base, {}).get("n_probes", "")
        elif base == "centroid_join":
            cj = stored_params.get("centroid_join", {})
            row["n_clusters"] = cj.get("n_clusters", "")
            row["k_db_candidates"] = cj.get("k_db_candidates", "")

        rows.append(row)

    # Read recall from statistics.json if it exists
    stats_path = os.path.join(output_dir, "statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats_list = json.load(f)
        recall_map = {s["method"]: s.get("recall") for s in stats_list}
        for row in rows:
            row["recall"] = recall_map.get(row["method_name"], "")

    csv_path = os.path.join(output_dir, "sweep_summary.csv")
    if rows:
        fieldnames = ["method_name", "base_method", "time_s", "pair_count",
                      "recall", "k", "itopk_size", "n_lists", "n_probes",
                      "n_clusters", "k_db_candidates"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n-> Sweep summary written to {csv_path}")


def main():
    # ── Load data ───────────────────────────────────────────────
    print("Loading dataset A...")
    dataset_A = load_dataset(DATASET_A_PATH, DATASET_A_FMT,
                             max_vectors=NUM_VECTORS_A)
    print(f"Dataset A: {dataset_A.shape[0]:,} x {dataset_A.shape[1]}D")

    self_join = not DATASET_B_PATH
    if self_join:
        print("Self-join mode (no dataset B specified)\n")
        dataset_B = dataset_A
    else:
        print("\nLoading dataset B...")
        dataset_B = load_dataset(DATASET_B_PATH, DATASET_B_FMT,
                                 max_vectors=NUM_VECTORS_B)
        print(f"Dataset B: {dataset_B.shape[0]:,} x {dataset_B.shape[1]}D\n")

    rng = np.random.default_rng(42)
    dataset_A = dataset_A[rng.permutation(len(dataset_A))]
    if not self_join:
        dataset_B = dataset_B[rng.permutation(len(dataset_B))]
    else:
        dataset_B = dataset_A
    print("Shuffled vectors (seed=42)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Ground truth (brute force) ──────────────────────────
    if _sweep_point_exists(OUTPUT_DIR, "brute_force"):
        print("\n[SKIP] brute_force already exists, reusing as ground truth")
    else:
        params = _make_params()
        _run_single(dataset_A, dataset_B, self_join, OUTPUT_DIR,
                    "brute_force", "brute_force", params)

    # ── 2. cuVS CAGRA sweep (k × itopk_size) ──────────────────
    from params import CuvsCagraParams
    for k, itopk in product(CUVS_CAGRA_K_SWEEP, CUVS_CAGRA_ITOPK_SWEEP):
        # itopk_size must be >= k; skip invalid combos
        if itopk < k:
            continue
        sweep_name = f"cuvs_cagra_k{k}_itopk{itopk}"
        if _sweep_point_exists(OUTPUT_DIR, sweep_name):
            print(f"\n[SKIP] {sweep_name} already exists")
            continue
        params = _make_params(cuvs_cagra=CuvsCagraParams(k=k, itopk_size=itopk))
        _run_single(dataset_A, dataset_B, self_join, OUTPUT_DIR,
                    sweep_name, "cuvs_cagra", params)

    # ── 3. cuVS IVF-Flat sweep (k × n_lists × n_probes) ──────
    from params import CuvsIvfFlatParams
    for k, nlists, nprobes in product(CUVS_IVF_FLAT_K_SWEEP,
                                       CUVS_IVF_FLAT_NLISTS_SWEEP,
                                       CUVS_IVF_FLAT_NPROBES_SWEEP):
        # n_probes > n_lists is pointless (would just be brute force)
        if nprobes > nlists:
            continue
        sweep_name = f"cuvs_ivf_flat_k{k}_nl{nlists}_np{nprobes}"
        if _sweep_point_exists(OUTPUT_DIR, sweep_name):
            print(f"\n[SKIP] {sweep_name} already exists")
            continue
        params = _make_params(cuvs_ivf_flat=CuvsIvfFlatParams(
            k=k, n_lists=nlists, n_probes=nprobes))
        _run_single(dataset_A, dataset_B, self_join, OUTPUT_DIR,
                    sweep_name, "cuvs_ivf_flat", params)

    # ── 4. Centroid join sweep (n_clusters × k_db_candidates) ─
    from params import CentroidJoinParams
    for nc, kdb in product(CENTROID_N_CLUSTERS_SWEEP,
                           CENTROID_K_DB_CANDIDATES_SWEEP):
        sweep_name = f"centroid_join_nc{nc}_kdb{kdb}"
        if _sweep_point_exists(OUTPUT_DIR, sweep_name):
            print(f"\n[SKIP] {sweep_name} already exists")
            continue
        params = _make_params(
            centroid_join=CentroidJoinParams(n_clusters=nc,
                                            k_db_candidates=kdb)
        )
        _run_single(dataset_A, dataset_B, self_join, OUTPUT_DIR,
                    sweep_name, "centroid_join", params)

    # ── 5. Compute recall + statistics ─────────────────────────
    print("\n\nComputing recall against brute_force ground truth...")
    stats = process_results(OUTPUT_DIR)
    dump_statistics(OUTPUT_DIR, stats)

    # ── 6. Generate tidy summary CSV ───────────────────────────
    generate_sweep_summary(OUTPUT_DIR)

    print("\n✓ Sweep complete!")


if __name__ == "__main__":
    main()
