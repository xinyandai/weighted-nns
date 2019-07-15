import numpy as np
import numba as nb
import math
import tqdm

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix,
        out=np.zeros_like(vecs), where=norms_matrix != 0)
    return norms, normalized_vecs

@nb.jit
def arg_sort(distances, top_k):
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]



@nb.jit
def weighted_euclidean(q, x, w, top_k):
    distances = np.sum(np.multiply((q - x)**2, w), axis=1)
    return arg_sort(distances, top_k)


@nb.jit
def euclidean(q, x, top_k):
    distances = np.linalg.norm((q - x), axis=1)
    return arg_sort(distances, top_k)


@nb.jit
def product_arg_sort(q, x, top_k):
    distances = np.dot(x, -q)
    return arg_sort(distances, top_k)


@nb.jit
def parallel_sort(x, q, w=None, metric="weighted"):
    """
    for each q in 'Q', sort the x items by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param x: x items, same dimension as origin data, shape(N * D)
    :param q: queries, shape(len(Q) * D)
    :return:
    """
    top_k = min(131072, x.shape[0]-1)
    rank = np.empty((q.shape[0], top_k), dtype=np.int32)
    p_range =tqdm.tqdm(nb.prange(q.shape[0]))

    if metric == 'weighted':
        for i in p_range:
            rank[i, :] = weighted_euclidean(q[i], x, w, top_k)
    elif metric == 'angular':
        _, normalized_x = normalize(x)
        for i in p_range:
            rank[i, :] = product_arg_sort(q[i], normalized_x, top_k)
    elif metric == "euclid":
        for i in p_range:
            rank[i, :] = euclidean(q[i], x, top_k)
    else:
        assert False
    return rank
