import numpy as np
from vec_io import fvecs_read
from sorter import parallel_sort
from lsh import SRP, E2LSH, MultiTable
from transform import transform, spherical_transform

def intersect(gs, ids):
    rc = np.mean([
        len(np.intersect1d(g, list(id)))
        for g, id in zip(gs, ids)])
    return rc


def recalls(index, q_, gt):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    print(" Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in 2**np.arange(0, 16):
        ids = index.search(q_, t)
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], ids)
            print("%.4f \t" % (rc / float(top_k)), end="")
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
    np.random.seed(808)
    x, q = load_data()
    n, d = x.shape

    w = np.random.uniform(size=d)
    w = w / np.linalg.norm(w)
    gt = parallel_sort(x, q, w, metric="weighted")

    print("=")
    print("=")


    print("==================ours_transform====================")
    x_, q_ = transform(x, q, w)
    n_, d_ = x_.shape
    index = SRP(k=256, d=d_)
    index.add(x_)
    recalls(index, q_, gt)
    print("==================spherical_transform====================")
    x_, q_ = spherical_transform(x, q, w)
    n_, d_ = x_.shape
    index = SRP(k=256, d=d_)
    index.add(x_)
    recalls(index, q_, gt)


    # gt = parallel_sort(x, q, metric="angular")
    # index = SRP(k=32, d=d)
    # index.add(x)
    # recalls(index, q, gt)

    # gt = parallel_sort(x, q, metric="euclid")
    # index = E2LSH(r=2.0, k=32, d=d)
    # index.add(x)
    # recalls(index, q, gt)


if __name__ == "__main__":
    main()
