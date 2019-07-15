import numpy as np
from lsh import SRP
from scipy.spatial.distance import cdist
from transform import scale_mean, simple_lsh
from lsh_spherical import load_data, parallel_sort, intersect


def transform(x, q, w, scalar=0.2):
    x, q = scale_mean(x, q, scalar=scalar)
    return simple_lsh(x**2, np.tile(- w, (len(q), 1))), \
           simple_lsh(x, 2.0 * np.multiply(w, q))


def _hamming_dist(q, x):
    return cdist(q, x, 'hamming')


def test_recalls(sort, gt):
    ks = [20, 100]
    ts = [16, 128, 1024]
    print(" Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in ts:
        print("%6d \t %6d \t" % (t, len(sort[0, :t])), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], sort[:, :t])
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()


def composite():
    x, q = load_data()
    n, d = x.shape

    np.random.seed(808)
    w = np.random.uniform(size=d)
    w = w / np.linalg.norm(w)

    gt = parallel_sort(x, q, w, metric="weighted")

    (x1, q1), (x2, q2) = transform(x, q, w)
    U1 = np.linalg.norm(x1[0, :])
    U2 = np.linalg.norm(x2[0, :])
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    for rate in [2,4,6,8,10]:
        ks = 256
        k1 = int(ks / rate)
        k2 = ks - k1
        print()
        print("k1=", k1, " k2=", k2)
        # np.random.seed(808)
        h1, h2 = SRP(k=k1, d=d1), SRP(k=k2, d=d2)
        xc1, xc2 = h1.hash(x1), h2.hash(x2)
        qc1, qc2 = h1.hash(q1), h2.hash(q2)
        l1, l2 = _hamming_dist(qc1, xc1), \
                 _hamming_dist(qc2, xc2)
        def estimate_similarity(u, l, k, alpha=1.0, shift=0.0):
            return u * np.cos(np.pi * (alpha * (l / k) + shift))
        similarity = estimate_similarity(U1, l1, k1,
                                         alpha=0.9, shift=0.1) + \
                     estimate_similarity(U2, l2, k2,
                                         alpha=0.9, shift=0.1)

        test_recalls(np.argsort(-similarity), gt)



if __name__ == "__main__":
    composite()
