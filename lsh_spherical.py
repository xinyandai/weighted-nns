import numpy as np
from vec_io import fvecs_read
from sorter import parallel_sort
from lsh import SRP
from transform import spherical_transform, simple_lsh


def intersect(gs, ids):
    rc = np.mean([
        len(np.intersect1d(g, list(id)))
        for g, id in zip(gs, ids)])
    return rc


def recalls(index, q_, gt):
    ks = [20, 100]
    ts = [16, 128, 1024]
    print(" Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in ts:
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
    # dataset = "imagenet"
    # base = "/research/jcheng2/xinyan/data/"
    x_path = "{}{}/{}_base.fvecs".format(base, dataset, dataset)
    q_path = "{}{}/{}_query.fvecs".format(base, dataset, dataset)
    x = fvecs_read(x_path)
    q = fvecs_read(q_path)[:1000]
    return x, q


def main():
    x, q = load_data()
    n, d = x.shape

    np.random.seed(808)
    w = np.random.uniform(size=d)
    w = w / np.linalg.norm(w)

    gt = parallel_sort(x, q, w, metric="weighted")

    ks = 256

    print("==================spherical_transform====================")
    x_, q_ = spherical_transform(x, q, w)
    n_, d_ = x_.shape
    np.random.seed(808)
    index = SRP(k=ks, d=d_)
    index.add(x_)
    recalls(index, q_, gt)


if __name__ == "__main__":
    main()
