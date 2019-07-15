import numpy as np


def scale_mean(x, q, scalar):
    mean = np.mean(x, axis=0, keepdims=True)
    x -= mean
    q -= mean
    scale = np.max(np.abs(x)) / scalar
    x /= scale
    q /= scale
    return x ,q


def spherical_transform(x, q, w, U=np.pi):
    x, q = scale_mean(x, q, scalar=U)

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
    x_[:, d] = np.sqrt(m**2 - norms**2)
    q_[:, d] = 0.0

    return x_, q_


def transform(x, q, w, scalar=0.2):
    x, q = scale_mean(x, q, scalar=scalar)

    x = np.hstack([x**2, x])
    q = np.hstack([
        np.tile(- w, (len(q), 1)), 2.0 * np.multiply(w, q)
    ])
    return simple_lsh(x, q)

