import cupy as cp
from cuvs.neighbors import ivf_flat, brute_force

# Small test
N, D = 10000, 128
vectors = cp.random.random((N, D), dtype=cp.float32)

# Build IVF index
build_params = ivf_flat.IndexParams(n_lists=100)
search_params = ivf_flat.SearchParams(n_probes=10)
ivf_index = ivf_flat.build(build_params, vectors)

# Query with IVF (k=10)
queries = vectors[:100]  # First 100 as queries
ivf_dists, ivf_neighbors = ivf_flat.search(search_params, ivf_index, queries, k=10)
ivf_neighbors = cp.asarray(ivf_neighbors)

# Query with brute force (k=10) as ground truth
bf_index = brute_force.build(vectors)
bf_dists, bf_neighbors = brute_force.search(bf_index, queries, k=10)
bf_neighbors = cp.asarray(bf_neighbors)

# Compute recall@10
recall = 0
for i in range(100):
    ivf_set = set(ivf_neighbors[i].get().tolist())
    bf_set = set(bf_neighbors[i].get().tolist())
    recall += len(ivf_set & bf_set) / 10

print(f"IVF recall@10 (n_lists=100, n_probes=10): {recall:.2%}")

# Also check recall@1 (top-1 accuracy)
recall_1 = (ivf_neighbors[:, 0] == bf_neighbors[:, 0]).sum().item() / 100
print(f"IVF recall@1: {recall_1:.2%}")
