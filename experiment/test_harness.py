import time
import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from brute_force_similarity import brute_force_threshold_join
from ivf_similarity import ivf_threshold_join
from centroid_similarity import run_threshold_similarity_join


@dataclass
class BenchmarkResult:
    name: str
    time_s: float
    pair_count: int
    pairs: List[Tuple[int, int, float]]


def load_fvecs(filename: str) -> cp.ndarray:
    """Load .fvecs format file."""
    with open(filename, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='float32')
        vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
        print(f"Loaded {vectors.shape[0]} x {vectors.shape[1]} from {filename}")
        return cp.asarray(vectors)


def load_bvecs(filename: str) -> cp.ndarray:
    """Load .bvecs format file."""
    with open(filename, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='uint8')
        vectors = x.reshape(-1, dim + 4)[:, 4:].astype(np.float32).copy()
        print(f"Loaded {vectors.shape[0]} x {vectors.shape[1]} from {filename}")
        return cp.asarray(vectors)


def generate_random(n: int, d: int, normalize: bool = True) -> cp.ndarray:
    """Generate random vectors."""
    vectors = cp.random.random((n, d), dtype=cp.float32)
    if normalize:
        vectors /= cp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def run_benchmark(
    vectors_A: cp.ndarray,
    vectors_B: cp.ndarray,
    threshold: float,
    self_join: bool = False,
    methods: Optional[List[str]] = None
) -> List[BenchmarkResult]:
    """
    Run similarity join benchmarks.
    
    Args:
        vectors_A: Database vectors (N x D)
        vectors_B: Query vectors (M x D)
        threshold: Distance threshold
        self_join: If True, exclude self-matches and (a,b)/(b,a) duplicates
        methods: List of methods to run. Default: all.
                 Options: 'brute_force', 'ivf', 'centroid'
    """
    if methods is None:
        methods = ['brute_force', 'ivf', 'centroid']
    
    results = []
    N, D = vectors_A.shape
    M = vectors_B.shape[0]
    print(f"\n{'='*60}")
    print(f"Benchmark: N={N}, M={M}, D={D}, threshold={threshold}")
    print(f"{'='*60}")
    
    if 'ivf' in methods:
        print("\n[1/3] IVF-Flat")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        pairs = ivf_threshold_join(
            vectors_A, vectors_B, threshold,
            n_lists=256,
            n_probes=32,
            k_candidates=256,
            batch_size=100_000,
            self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"IVF finished in {elapsed:.4f}s found: {len(pairs)} pairs")
        results.append(BenchmarkResult('ivf', elapsed, len(pairs), pairs))
    
    if 'centroid' in methods:
        print("\n[2/3] Centroid Clustering")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        pairs = run_threshold_similarity_join(
            vectors_A, vectors_B, threshold,
            n_clusters=256,
            k_db_candidates=2048,
            batch_size=100_000,
            self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"Centroid finished in {elapsed:.4f}s found: {len(pairs)} pairs")
        results.append(BenchmarkResult('centroid', elapsed, len(pairs), pairs))

    if 'brute_force' in methods:
        print("\n[3/3] Brute Force")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        pairs = brute_force_threshold_join(
            vectors_A, vectors_B, threshold,
            batch_size=10_000, self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"Brute force finished in {elapsed:.4f}s found: {len(pairs)} pairs")
        results.append(BenchmarkResult('brute_force', elapsed, len(pairs), pairs))
    
    return results


def compare_results(results: List[BenchmarkResult], ground_truth_name: str = 'brute_force'):
    """Compare results against ground truth."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    gt = next((r for r in results if r.name == ground_truth_name), None)
    gt_set = set((a, b) for a, b, _ in gt.pairs) if gt else None
    
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Time: {r.time_s:.4f}s")
        print(f"  Pairs: {r.pair_count}")
        
        if r.pairs:
            dists = [d for _, _, d in r.pairs]
            print(f"  Dists: min={min(dists):.5f} max={max(dists):.5f} avg={sum(dists)/len(dists):.5f}")
        
        if gt and r.name != ground_truth_name and gt_set:
            r_set = set((a, b) for a, b, _ in r.pairs)
            recall = len(r_set & gt_set) / len(gt_set) if gt_set else 0
            print(f"  Recall vs {ground_truth_name}: {recall:.4f}")


def load_fvecs(path: str) -> np.ndarray:
    """Load .fvecs format file."""
    with open(path, 'rb') as f:
        # Read dimensions
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        # Read entire file as float32 (since int32 and float32 are same size)
        x = np.fromfile(f, dtype='float32')
        # Reshape to (N, dim+1) where +1 is the dimension header
        vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
    return vectors


def load_sift(max_vectors: int = None) -> cp.ndarray:
    """Load SIFT-1M base vectors."""
    path = "/home/william/thesis_ws/thesis-repo/datasets/sift/sift_base.fvecs"
    vectors = load_fvecs(path)
    if max_vectors:
        vectors = vectors[:max_vectors]
    return cp.asarray(vectors, dtype=cp.float32)


def main():
    print("Loading SIFT data...")
    vectors = load_sift(1_000_000)
    print(f"Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    
    # Use a threshold that gives reasonable pair count
    threshold = 30000.0  # squared L2, based on sample output (mean~26k)
    
    results = run_benchmark(
        vectors_A=vectors,
        vectors_B=vectors,
        threshold=threshold,
        self_join=True,
        methods=['ivf', 'centroid', 'brute_force']
    )
    
    compare_results(results)
    
    # Show top pairs from each method
    for r in results:
        if r.pairs:
            print(f"\n{r.name} top 5 pairs:")
            for a, b, d in sorted(r.pairs, key=lambda x: x[2])[:5]:
                print(f"  A[{a}] <-> B[{b}] dist={d:.5f}")


if __name__ == "__main__":
    main()
