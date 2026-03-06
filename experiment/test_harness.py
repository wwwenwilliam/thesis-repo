import time
import cupy as cp
import numpy as np

from brute_force_similarity import brute_force_threshold_join
from ivf_similarity import ivf_threshold_join
from centroid_similarity import run_threshold_similarity_join
from result_store import ResultStore


class BenchmarkResult:
    def __init__(self, name, time_s, pair_count, pairs_a=None, pairs_b=None, pairs_dist=None, result_set=None):
        self.name = name
        self.time_s = time_s
        self.pair_count = pair_count
        self.pairs_a = pairs_a
        self.pairs_b = pairs_b
        self.pairs_dist = pairs_dist
        self.result_set = result_set   # disk-backed results (set when output_dir used)

    @property
    def is_disk_backed(self):
        return self.result_set is not None


def load_fvecs(filename):
    """Load .fvecs format file."""
    with open(filename, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='float32')
        vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
        print(f"Loaded {vectors.shape[0]} x {vectors.shape[1]} from {filename}")
        return cp.asarray(vectors)


def load_bvecs(filename):
    """Load .bvecs format file."""
    with open(filename, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='uint8')
        vectors = x.reshape(-1, dim + 4)[:, 4:].astype(np.float32).copy()
        print(f"Loaded {vectors.shape[0]} x {vectors.shape[1]} from {filename}")
        return cp.asarray(vectors)


def generate_random(n, d, normalize=True):
    """Generate random vectors."""
    vectors = cp.random.random((n, d), dtype=cp.float32)
    if normalize:
        vectors /= cp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def run_benchmark(vectors_A, vectors_B, threshold, self_join=False, methods=None, output_dir=None):
    """
    Run similarity join benchmarks.
    
    Args:
        vectors_A: Database vectors (N x D)
        vectors_B: Query vectors (M x D)
        threshold: Distance threshold
        self_join: If True, exclude self-matches and (a,b)/(b,a) duplicates
        methods: List of methods to run. Default: all.
                 Options: 'brute_force', 'ivf', 'centroid'
        output_dir: If set, save results to disk after timing (does not affect benchmark)
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
            self_join=self_join,
            return_chunks=bool(output_dir)
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        if output_dir:
            n_pairs = sum(len(c) for c in a_idx)
            print(f"Centroid finished in {elapsed:.4f}s found: {n_pairs} pairs")
            rs = ResultStore.save_chunks(output_dir, 'centroid', a_idx, b_idx, dists,
                                         metadata={'time_s': elapsed, 'threshold': threshold})
            results.append(BenchmarkResult('centroid', elapsed, n_pairs, result_set=rs))
        else:
            n_pairs = len(a_idx)
            print(f"Centroid finished in {elapsed:.4f}s found: {n_pairs} pairs")
            results.append(BenchmarkResult('centroid', elapsed, n_pairs, a_idx, b_idx, dists))
    
    if 'ivf' in methods:
        print("\nIVF-Flat")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        a_idx, b_idx, dists = ivf_threshold_join(
            vectors_A, vectors_B, threshold,
            n_lists=256,
            n_probes=64,
            k_candidates=1024,
            batch_size=250_000,
            self_join=self_join,
            prebuild_blocks=False,
            return_chunks=bool(output_dir)
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        if output_dir:
            n_pairs = sum(len(c) for c in a_idx)
            print(f"IVF finished in {elapsed:.4f}s found: {n_pairs} pairs")
            rs = ResultStore.save_chunks(output_dir, 'ivf', a_idx, b_idx, dists,
                                         metadata={'time_s': elapsed, 'threshold': threshold})
            results.append(BenchmarkResult('ivf', elapsed, n_pairs, result_set=rs))
        else:
            n_pairs = len(a_idx)
            print(f"IVF finished in {elapsed:.4f}s found: {n_pairs} pairs")
            results.append(BenchmarkResult('ivf', elapsed, n_pairs, a_idx, b_idx, dists))

    if 'brute_force' in methods:
        print("\nBrute Force")
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        a_idx, b_idx, dists = brute_force_threshold_join(
            vectors_A, vectors_B, threshold,
            batch_size=20_000, self_join=self_join,
            return_chunks=bool(output_dir)
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        if output_dir:
            n_pairs = sum(len(c) for c in a_idx)
            print(f"Brute force finished in {elapsed:.4f}s found: {n_pairs} pairs")
            rs = ResultStore.save_chunks(output_dir, 'brute_force', a_idx, b_idx, dists,
                                          metadata={'time_s': elapsed, 'threshold': threshold})
            results.append(BenchmarkResult('brute_force', elapsed, n_pairs, result_set=rs))
        else:
            n_pairs = len(a_idx)
            print(f"Brute force finished in {elapsed:.4f}s found: {n_pairs} pairs")
            results.append(BenchmarkResult('brute_force', elapsed, n_pairs, a_idx, b_idx, dists))
    
    return results


def _load_pairs(r):
    """Get arrays from a BenchmarkResult, whether in-memory or disk-backed."""
    if r.is_disk_backed:
        return r.result_set.load_all()
    return r.pairs_a, r.pairs_b, r.pairs_dist


def _build_pair_set(r):
    """Build a set of (a, b) pairs, using chunked iteration for disk-backed results."""
    if r.is_disk_backed:
        pair_set = set()
        for a_chunk, b_chunk, _ in r.result_set.iter_chunks():
            pair_set.update(zip(a_chunk.tolist(), b_chunk.tolist()))
        return pair_set
    return set(zip(r.pairs_a.tolist(), r.pairs_b.tolist()))


def compare_results(results, ground_truth_name='brute_force'):
    """Compare results against ground truth."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    gt = next((r for r in results if r.name == ground_truth_name), None)
    gt_set = _build_pair_set(gt) if gt and gt.pair_count > 0 else None
    
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Time: {r.time_s:.4f}s")
        print(f"  Pairs: {r.pair_count}")
        
        if r.pair_count > 0:
            _, _, dists = _load_pairs(r)
            print(f"  Dists: min={dists.min():.5f} max={dists.max():.5f} avg={dists.mean():.5f}")
        
        if gt and r.name != ground_truth_name and gt_set and r.pair_count > 0:
            r_set = _build_pair_set(r)
            recall = len(r_set & gt_set) / len(gt_set) if gt_set else 0
            print(f"  Recall vs {ground_truth_name}: {recall:.4f}")


def load_fvecs_np(path):
    """Load .fvecs format file as numpy array."""
    with open(path, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='float32')
        vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
    return vectors


def load_u8bin(path, max_vectors=None):
    """Load .u8bin format file (8-byte header + row-major uint8 data)."""
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        d = int.from_bytes(f.read(4), 'little')
    print(f"u8bin header: {n:,} vectors x {d}D")
    if max_vectors:
        n = min(n, max_vectors)
    data = np.memmap(path, dtype='uint8', mode='r', offset=8, shape=(n, d))
    vectors = np.array(data[:n], dtype=np.float32)
    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]} from {path}")
    return vectors


def load_sift(max_vectors=None, gpu=True):
    """Load SIFT-1M base vectors. Set gpu=False for out-of-memory workflows."""
    path = "/home/william/thesis_ws/thesis-repo/datasets/sift/sift_base.fvecs"
    vectors = load_fvecs_np(path)
    if max_vectors:
        vectors = vectors[:max_vectors]
    if gpu:
        return cp.asarray(vectors, dtype=cp.float32)
    return vectors.astype(np.float32)


def load_sift_100m(max_vectors=None, gpu=False):
    """Load SIFT-100M from u8bin. Default gpu=False since 100M x 128 x 4 = 51GB won't fit in VRAM."""
    path = "/home/william/thesis_ws/datasets/sift/learn.100M.u8bin"
    vectors = load_u8bin(path, max_vectors=max_vectors)
    if gpu:
        return cp.asarray(vectors, dtype=cp.float32)
    return vectors


def load_results(results_dir):
    """Load and compare previously saved results without re-running."""
    print(f"Loading results from {results_dir}")
    runs = ResultStore.list_runs(results_dir)
    print(f"Found runs: {runs}")
    results = []
    for name in runs:
        rs = ResultStore.load(f"{results_dir}/{name}")
        meta = rs.metadata
        results.append(BenchmarkResult(
            name=name,
            time_s=meta.get('time_s', 0),
            pair_count=rs.pair_count,
            result_set=rs
        ))
    compare_results(results)
    return results


# ── Configuration ───────────────────────────────────────────────
# Edit these directly to change benchmark parameters.

N_VECTORS   = 10_000_000
THRESHOLD   = 20_000.0
# METHODS     = ['centroid', 'ivf', 'brute_force']
METHODS     = ['brute_force']
OUTPUT_DIR  = "./results"       # set to None to keep results in RAM (may OOM at scale)


def main():
    print("Loading SIFT data...")
    vectors = load_sift_100m(N_VECTORS)
    print(f"Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    
    # Shuffle to break sequential ordering — ensures each IVF block
    # contains a representative sample, not a localized region.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(vectors))
    vectors = vectors[perm]
    print("Shuffled vectors (seed=42)")
    
    results = run_benchmark(
        vectors_A=vectors,
        vectors_B=vectors,
        threshold=THRESHOLD,
        self_join=True,
        methods=METHODS,
        output_dir=OUTPUT_DIR
    )
    
    compare_results(results)
    
    # Show top pairs from each method
    for r in results:
        if r.pair_count > 0:
            a, b, d = _load_pairs(r)
            top_idx = np.argsort(d)[:5]
            print(f"\n{r.name} top 5 pairs:")
            for i in top_idx:
                print(f"  A[{a[i]}] <-> B[{b[i]}] dist={d[i]:.5f}")


if __name__ == "__main__":
    main()
