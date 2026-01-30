"""
Benchmark batching overhead for IVF-Flat search.
Measures time spent on:
1. GPU compute (search)
2. Memory transfer (cp.asnumpy, concatenation)
3. Top-K selection (argpartition, argsort)
"""
import time
import cupy as cp
from cuvs.neighbors import ivf_flat

def read_fvecs(filename: str) -> cp.ndarray:
    import numpy as np
    with open(filename, 'rb') as f:
        f.read(4)
        f.seek(0)
        x = np.fromfile(f, dtype='float32')
        vectors = x.reshape(-1, 129)[:, 1:].copy()
        return cp.asarray(vectors)

print("Loading SIFT1M...")
vectors = read_fvecs('/home/william/thesis_ws/datasets/sift/sift_base.fvecs')
N = vectors.shape[0]
print(f"Loaded {N:,} vectors\n")

# Build index once
build_params = ivf_flat.IndexParams(n_lists=4096)
index = ivf_flat.build(build_params, vectors)
search_params = ivf_flat.SearchParams(n_probes=64)

k_global = 10_000
k_per_query = 100
batch_size = 50_000

# Timing accumulators
time_search = 0
time_memcpy = 0
time_topk = 0

best_dists = cp.full(k_global, cp.inf, dtype=cp.float32)
best_a = cp.zeros(k_global, dtype=cp.int64)
best_b = cp.zeros(k_global, dtype=cp.int64)

t_total_start = time.perf_counter()

for b_start in range(0, N, batch_size):
    b_end = min(b_start + batch_size, N)
    batch = vectors[b_start:b_end]
    
    # 1. GPU Search
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    distances, neighbors = ivf_flat.search(search_params, index, batch, k=k_per_query)
    cp.cuda.Stream.null.synchronize()
    time_search += time.perf_counter() - t0
    
    # 2. Memory operations (GPU arrays + transfers)
    t0 = time.perf_counter()
    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)
    distances[distances <= 1e-4] = cp.inf
    
    flat_distances = distances.flatten()
    flat_neighbors = neighbors.flatten()
    flat_b_indices = cp.repeat(cp.arange(b_start, b_end, dtype=cp.int64), k_per_query)
    
    all_d = cp.concatenate([best_dists, flat_distances])
    all_a = cp.concatenate([best_a, flat_neighbors])
    all_b = cp.concatenate([best_b, flat_b_indices])
    cp.cuda.Stream.null.synchronize()
    time_memcpy += time.perf_counter() - t0
    
    # 3. Top-K selection
    t0 = time.perf_counter()
    top_idx = cp.argpartition(all_d, k_global)[:k_global]
    top_idx = top_idx[cp.argsort(all_d[top_idx])]
    
    best_dists = all_d[top_idx]
    best_a = all_a[top_idx]
    best_b = all_b[top_idx]
    cp.cuda.Stream.null.synchronize()
    time_topk += time.perf_counter() - t0
    
    del distances, neighbors
    cp.get_default_memory_pool().free_all_blocks()

time_total = time.perf_counter() - t_total_start

# Results
print("=" * 50)
print("Batching Overhead Analysis")
print("=" * 50)
print(f"Batch size: {batch_size:,}, k_per_query: {k_per_query}")
print(f"Number of batches: {N // batch_size}")
print()
print(f"{'Operation':<25} {'Time (s)':<12} {'% of Total':<12}")
print("-" * 50)
print(f"{'GPU Search':<25} {time_search:<12.4f} {100*time_search/time_total:<12.1f}%")
print(f"{'Memory Ops':<25} {time_memcpy:<12.4f} {100*time_memcpy/time_total:<12.1f}%")
print(f"{'Top-K Selection':<25} {time_topk:<12.4f} {100*time_topk/time_total:<12.1f}%")
print("-" * 50)
print(f"{'TOTAL':<25} {time_total:<12.4f}")
print()
print(f"Batching overhead (non-search): {100*(time_memcpy + time_topk)/time_total:.1f}%")
