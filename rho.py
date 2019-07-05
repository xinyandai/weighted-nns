import numpy as np
from math import *


def phi(x):
    """
    :return: Cumulative distribution function for the standard
             normal distribution
    """
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def prob_l2(d, r):
    x = float(r) / float(d)
    y = 2.0 / (sqrt(2.0 * pi) * x) * (1 - exp( - x * x / 2.0))
    return 1.0 - 2.0 * phi(-x) - y


def prob_srp(cosine):
    if cosine >= 1 or cosine <= -1:
        return nan
    return 1.0 - acos(cosine) / pi


def rho_sl(w_, mu, R_1, R_2, U, r):
    t = 2.0 * mu - 2.0 * (1 + 2 * w_ )
    a = t  + R_1 - 1.0 / 12 * w_ * (U**4)
    b = t + R_2 -  1.0 / 12 * (w_ + 1) * (U**4)
    if a < 0 or  b <0:
        return nan
    return log(prob_l2(sqrt(a), r)) / log(prob_l2(sqrt(b), r))


def rho_ss(w_, mu, R_1, R_2, U):
    a = 1.0 + 2.0 * w_ - 0.5 * R_1 + 1.0 / 24 * w_ * (U**4)
    b = 1.0 + 2.0 * w_ - 0.5 * R_2 + 1.0 / 24 * (w_ + 1) * (U**4)
    a /= mu
    b /= mu
    return log(prob_srp(a)) / log(prob_srp(b))


def min_rho_sl(w_, mu, R_1, R_2):
    min = max(sqrt(R_2 / (1.0 + w_)), sqrt(- R_1 / w_))
    Us = np.linspace(min, 10, 100)
    Rs =  2 ** np.linspace(-4, 10, 100)
    rhos = [[rho_sl(w_, mu, R_1, R_2, U, r)
             for r in Rs] for U in Us]
    return np.nanmin(rhos)


def min_rho_ss(w_, mu, R_1, R_2):
    min = max(sqrt(R_2 / (1.0 + w_)), sqrt(- R_1 / w_))
    Us = np.linspace(min, 10, 100)
    rhos = [rho_ss(w_, mu, R_1, R_2, U) for U in Us]
    return np.nanmin(rhos)


print(min_rho_sl(-0.5, 1.5, 0.2, 0.4))
print(min_rho_ss(-0.5, 1.5, 0.2, 0.4))

print(min_rho_sl(-0.5, 1.5, 0.3, 0.4))
print(min_rho_ss(-0.5, 1.5, 0.3, 0.4))

print(min_rho_sl(-0.5, 1.5, 0.5, 0.6))
print(min_rho_ss(-0.5, 1.5, 0.5, 0.6))