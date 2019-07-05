import numpy as np
from vec_io import fvecs_read
from sorter import parallel_sort
from lsh import SRP, E2LSH, MultiTable


def spherical_transform(x, q, w, U=np.pi):
    max = np.max(np.abs(x))
    x = x / max * U
    q = q / max * U

    x = np.hstack([np.cos(x), np.sin(x)])
    w = w[None, :]
    q = np.hstack([np.multiply(w, np.cos(q)),
                   np.multiply(w, np.sin(q))])
    return x, q


def simple_lsh(x, q):

    nx, d = np.shape(x)
    nq, _ = np.shape(q)
    x_ = np.empty(shape=(nx, d + 1))
    q_ = np.empty(shape=(nq, d + 1))

    x_[:, :d] = x
    q_[:, :d] = q

    norms = np.linalg.norm(x, axis=1)
    m = np.max(norms)
    x_[:, d] = m - norms
    q_[:, d] = 0.0

    return x_, q_


def transform(x, q, w):
    if x is not None:
        x = np.hstack([x**2, x])
    if q is not None and w is not None:
        q = np.hstack([np.tile(w, (len(q), 1)),
                       -2.0 * np.multiply(w, q)])
    return simple_lsh(x, q)


def intersect(gs, ids):
    rc = np.mean([
        len(np.intersect1d(g, list(id)))
        for g, id in zip(gs, ids)])
    return rc


def recalls(index, q_, gt):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    print("\t\t\t\t", end="")
    for top_k in ks:
        print("%4d\t" % (top_k), end="")
    print()
    for t in 2**np.arange(0, 16):
        ids = index.search(q_, t)
        items = np.mean([len(id) for id in ids])
        print("%4d \t %4d \t" % (t, items), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], ids)
            print("%.2f \t" % (rc / float(top_k)), end="")
        print()


def load_data():
    dataset = "netflix"
    base = "/home/xinyan/program/data/"
    x_path = "{}{}/{}_base.fvecs".format(base, dataset, dataset)
    q_path = "{}{}/{}_query.fvecs".format(base, dataset, dataset)
    x = fvecs_read(x_path)
    q = fvecs_read(q_path)[:100]
    return x, q


def main():
    x, q = load_data()
    n, d = x.shape

    w = np.random.normal(size=d)
    w = w / np.linalg.norm(w)
    gt = parallel_sort(x, q, w)
    x_, q_ = spherical_transform(x, q, w)
    n_, d_ = x_.shape
    index = SRP(k=256, d=d_)
    index.add(x_)
    recalls(index, q_, gt)

    # gt = parallel_sort(x, q, metric="angular")
    # index = SRP(k=32, d=d)
    # index.add(x)
    # recalls(index, q, gt)


main()
