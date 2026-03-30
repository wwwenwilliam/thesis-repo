import cupy as cp
from cupyx.profiler import time_range
import numpy as np
import rmm
from cuvs.neighbors import cagra, ivf_flat, brute_force


def _build_index(index_type, dataset, params):
    if index_type == 'cagra':
        idx_params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent")
        return cagra.build(idx_params, dataset)
    elif index_type == 'ivf_flat':
        idx_params = ivf_flat.IndexParams(metric="sqeuclidean")
        return ivf_flat.build(idx_params, dataset)
    elif index_type == 'brute_force':
        return brute_force.build(dataset, metric="sqeuclidean")
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _search_index(index_type, index, queries, params):
    if index_type == 'cagra':
        k = params.cuvs_cagra.k
        return cagra.search(cagra.SearchParams(), index, queries, k)
    elif index_type == 'ivf_flat':
        k = params.cuvs_ivf_flat.k
        return ivf_flat.search(ivf_flat.SearchParams(), index, queries, k)
    elif index_type == 'brute_force':
        k = params.cuvs_brute_force.k
        return brute_force.search(index, queries, k)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, index_type, params):
    if len(chunk_A) == 0 or len(chunk_B) == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                {})

    with time_range(f"cuvs_knn/{index_type}", color_id=0):
        gpu_A = cp.asarray(chunk_A)
        if self_join_diagonal:
            gpu_B = gpu_A
        else:
            gpu_B = cp.asarray(chunk_B)

        import time
        with time_range("cuvs_knn/build_index", color_id=1):
            cp.cuda.Stream.null.synchronize()
            t0 = time.time()
            index = _build_index(index_type, gpu_A, params)
            cp.cuda.Stream.null.synchronize()
            build_time = time.time() - t0

        with time_range("cuvs_knn/search", color_id=2):
            distances, indices = _search_index(index_type, index, gpu_B, params)
            distances = cp.asarray(distances)
            indices = cp.asarray(indices)

        with time_range("cuvs_knn/filter", color_id=3):
            mask = distances <= threshold

            nB = gpu_B.shape[0]
            b_idx_matrix = cp.arange(nB, dtype=cp.int64)[:, None]
            b_idx_matrix = cp.broadcast_to(b_idx_matrix, indices.shape)

            b_local = b_idx_matrix[mask]
            a_local = indices[mask]
            result_dists = distances[mask]

            if self_join_diagonal:
                diag_mask = a_local > b_local
                b_local = b_local[diag_mask]
                a_local = a_local[diag_mask]
                result_dists = result_dists[diag_mask]

            pairs_compared = distances.size

        with time_range("cuvs_knn/collect_results", color_id=4):
            return (a_local.get().astype(np.int64),
                    b_local.get().astype(np.int64),
                    result_dists.get().astype(np.float32),
                    {"build_time_s": build_time, "pairs_compared": int(pairs_compared)})


def cuvs_cagra_join(chunk_A, chunk_B, threshold, self_join_diagonal=False, params=None):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'cagra', params)


def cuvs_ivf_flat_join(chunk_A, chunk_B, threshold, self_join_diagonal=False, params=None):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'ivf_flat', params)

def cuvs_brute_force_join(chunk_A, chunk_B, threshold, self_join_diagonal=False, params=None):
    return _cuvs_knn_join(chunk_A, chunk_B, threshold, self_join_diagonal, 'brute_force', params)
