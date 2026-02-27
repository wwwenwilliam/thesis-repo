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
    pairs_a: np.ndarray   # array of A indices
    pairs_b: np.ndarray   # array of B indices
    pairs_dist: np.ndarray  # array of distances


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

    if 'centroid' in methods:
        print("\nCentroid Clustering")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        a_idx, b_idx, dists = run_threshold_similarity_join(
            vectors_A, vectors_B, threshold,
            n_clusters=1024,
            k_db_candidates=8192,
            batch_size=1_000_000,
            self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        n_pairs = len(a_idx)
        print(f"Centroid finished in {elapsed:.4f}s found: {n_pairs} pairs")
        results.append(BenchmarkResult('centroid', elapsed, n_pairs, a_idx, b_idx, dists))
    
    if 'ivf' in methods:
        print("\nIVF-Flat")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        a_idx, b_idx, dists = ivf_threshold_join(
            vectors_A, vectors_B, threshold,
            n_lists=512,
            n_probes=32,
            k_candidates=512,
            batch_size=250_000,
            self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        n_pairs = len(a_idx)
        print(f"IVF finished in {elapsed:.4f}s found: {n_pairs} pairs")
        results.append(BenchmarkResult('ivf', elapsed, n_pairs, a_idx, b_idx, dists))

    if 'brute_force' in methods:
        print("\nBrute Force")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        a_idx, b_idx, dists = brute_force_threshold_join(
            vectors_A, vectors_B, threshold,
            batch_size=20_000, self_join=self_join
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        n_pairs = len(a_idx)
        print(f"Brute force finished in {elapsed:.4f}s found: {n_pairs} pairs")
        results.append(BenchmarkResult('brute_force', elapsed, n_pairs, a_idx, b_idx, dists))
    
    return results


def compare_results(results: List[BenchmarkResult], ground_truth_name: str = 'brute_force'):
    """Compare results against ground truth."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    gt = next((r for r in results if r.name == ground_truth_name), None)
    gt_set = set(zip(gt.pairs_a.tolist(), gt.pairs_b.tolist())) if gt and gt.pair_count > 0 else None
    
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Time: {r.time_s:.4f}s")
        print(f"  Pairs: {r.pair_count}")
        
        if r.pair_count > 0:
            print(f"  Dists: min={r.pairs_dist.min():.5f} max={r.pairs_dist.max():.5f} avg={r.pairs_dist.mean():.5f}")
        
        if gt and r.name != ground_truth_name and gt_set and r.pair_count > 0:
            r_set = set(zip(r.pairs_a.tolist(), r.pairs_b.tolist()))
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


def load_u8bin(path: str, max_vectors: int = None) -> np.ndarray:
    """Load .u8bin format file (8-byte header + row-major uint8 data)."""
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        d = int.from_bytes(f.read(4), 'little')
    print(f"u8bin header: {n:,} vectors x {d}D")
    if max_vectors:
        n = min(n, max_vectors)
    # Memory-map to avoid loading entire file into RAM
    data = np.memmap(path, dtype='uint8', mode='r', offset=8, shape=(n, d))
    vectors = np.array(data[:n], dtype=np.float32)
    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]} from {path}")
    return vectors


def load_sift(max_vectors: int = None, gpu: bool = True) -> cp.ndarray:
    """Load SIFT-1M base vectors. Set gpu=False for out-of-memory workflows."""
    path = "/home/william/thesis_ws/thesis-repo/datasets/sift/sift_base.fvecs"
    vectors = load_fvecs(path)
    if max_vectors:
        vectors = vectors[:max_vectors]
    if gpu:
        return cp.asarray(vectors, dtype=cp.float32)
    return vectors.astype(np.float32)


def load_sift_100m(max_vectors: int = None, gpu: bool = False) -> np.ndarray:
    """Load SIFT-100M from u8bin. Default gpu=False since 100M x 128 x 4 = 51GB won't fit in VRAM."""
    path = "/home/william/thesis_ws/datasets/sift/learn.100M.u8bin"
    vectors = load_u8bin(path, max_vectors=max_vectors)
    if gpu:
        return cp.asarray(vectors, dtype=cp.float32)
    return vectors


def main():
    print("Loading SIFT data...")
    vectors = load_sift_100m(1_000_000)
    print(f"Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    
    # Use a threshold that gives reasonable pair count
    threshold = 20_000.0
    
    results = run_benchmark(
        vectors_A=vectors,
        vectors_B=vectors,
        threshold=threshold,
        self_join=True,
        methods=['centroid', 'ivf', 'brute_force']
    )
    
    compare_results(results)
    
    # Show top pairs from each method
    for r in results:
        if r.pair_count > 0:
            top_idx = np.argsort(r.pairs_dist)[:5]
            print(f"\n{r.name} top 5 pairs:")
            for i in top_idx:
                print(f"  A[{r.pairs_a[i]}] <-> B[{r.pairs_b[i]}] dist={r.pairs_dist[i]:.5f}")


if __name__ == "__main__":
    main()
