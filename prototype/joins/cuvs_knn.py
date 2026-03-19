import cupy as cp
from cupyx.profiler import time_range
import numpy as np
from cuvs.neighbors import cagra, ivf_pq, ivf_flat, brute_force

# Configuration: which index to use and its parameters
# Options: 'cagra', 'ivf_pq', 'ivf_flat', 'brute_force'
CUVS_INDEX_TYPE = 'brute_force'

def _build_index(index_type, dataset):
    if index_type == 'cagra':
        # We use nn_descent because the default IVF-PQ-based graph builder sometimes produces duplicate nodes
        # leading to "Could not generate an intermediate CAGRA graph" errors on subsets of SIFT.
        params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent")
        return cagra.build(params, dataset)
    elif index_type == 'ivf_pq':
        # Can adjust n_lists, pq_dim, etc.
        params = ivf_pq.IndexParams(metric="sqeuclidean")
        return ivf_pq.build(params, dataset)
    elif index_type == 'ivf_flat':
        params = ivf_flat.IndexParams(metric="sqeuclidean")
        return ivf_flat.build(params, dataset)
    elif index_type == 'brute_force':
        return brute_force.build(dataset, metric="sqeuclidean")
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _search_index(index_type, index, queries):
    if index_type == 'cagra':
        return cagra.search(cagra.SearchParams(), index, queries, 64)
    elif index_type == 'ivf_pq':
        return ivf_pq.search(ivf_pq.SearchParams(), index, queries, 512)
    elif index_type == 'ivf_flat':
        return ivf_flat.search(ivf_flat.SearchParams(), index, queries, 512)
    elif index_type == 'brute_force':
        return brute_force.search(index, queries, 512)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, index_type):
    if len(chunk_A) == 0 or len(chunk_B) == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                {})

    with time_range(f"cuvs_knn/{index_type}", color_id=0):
        gpu_A = cp.asarray(chunk_A)
        # Check if self_join_diagonal and same sizes - if so, it's the exact same data
        if self_join_diagonal:
            gpu_B = gpu_A
        else:
            gpu_B = cp.asarray(chunk_B)

        # Build the index on A
        import time
        with time_range("cuvs_knn/build_index", color_id=1):
            cp.cuda.Stream.null.synchronize()
            t0 = time.time()
            index = _build_index(index_type, gpu_A)
            cp.cuda.Stream.null.synchronize()
            build_time = time.time() - t0

        # Search queries B in index A
        with time_range("cuvs_knn/search", color_id=2):
            distances, indices = _search_index(index_type, index, gpu_B)

            # distances is shape (nB, k), indices is shape (nB, k)
            # The arrays returned by cuVS might be device_ndarrays, we need to convert them to cupy arrays.
            distances = cp.asarray(distances)
            indices = cp.asarray(indices)

        # Filter by threshold
        with time_range("cuvs_knn/filter", color_id=3):
            mask = distances <= threshold

            # Create b_indices using repeat/tile equivalent logic
            nB = gpu_B.shape[0]
            b_idx_matrix = cp.arange(nB, dtype=cp.int64)[:, None]
            b_idx_matrix = cp.broadcast_to(b_idx_matrix, indices.shape)

            # Apply mask
            b_local = b_idx_matrix[mask]
            a_local = indices[mask]
            result_dists = distances[mask]

            # self-join diagonal filtering
            if self_join_diagonal:
                diag_mask = a_local > b_local
                b_local = b_local[diag_mask]
                a_local = a_local[diag_mask]
                result_dists = result_dists[diag_mask]

            # The number of pairs subjected to the threshold filter is the size of the distances matrix (nB * k)
            pairs_compared = distances.size

        # Convert to numpy
        with time_range("cuvs_knn/collect_results", color_id=4):
            return (a_local.get().astype(np.int64),
                    b_local.get().astype(np.int64),
                    result_dists.get().astype(np.float32),
                    {"build_time_s": build_time, "pairs_compared": int(pairs_compared)})


def cuvs_cagra_join(chunk_A, chunk_B, threshold, self_join_diagonal=False):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'cagra')

def cuvs_ivf_pq_join(chunk_A, chunk_B, threshold, self_join_diagonal=False):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'ivf_pq')

def cuvs_ivf_flat_join(chunk_A, chunk_B, threshold, self_join_diagonal=False):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'ivf_flat')

def cuvs_brute_force_join(chunk_A, chunk_B, threshold, self_join_diagonal=False):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'brute_force')
