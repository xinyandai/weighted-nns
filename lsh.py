import numpy as np
from abc import ABC, abstractmethod


def _collect(ind, items, T):
    re = set([])
    for i in ind:
        re.update(items[i][1])
        if len(re) >= T:
            return re
    return re


class Index(ABC):
    def __init__(self, d):
        self.tables = {}
        self.items = None
        self.count = 0
        self.d = d

    def dist(self, hashes):
        self.items = list(self.tables.items())
        dist_ = np.zeros((len(hashes), len(self.items)),
                         dtype=np.int)
        for i, (key, value) in enumerate(self.items):
            key = np.reshape(key, (1, -1))
            dist_[:, i] = np.linalg.norm(
                hashes - key, ord=1, axis=1)
        return dist_

    @abstractmethod
    def hash(self, x):
        pass

    def get(self, hash_val):
        hash_val = tuple(hash_val)
        if hash_val not in self.tables:
            self.tables[hash_val] = []
        return self.tables[hash_val]

    def add(self, x: np.ndarray):
        assert x.shape[1] == self.d, \
            "shape {} is not matched with {}"\
            .format(x.shape, self.d)

        hashes = self.hash(x)
        for h in hashes:
            self.get(h).append(self.count)
            self.count += 1

    def search(self, q, T):
        hashes = self.hash(q)
        dist = self.dist(hashes)
        indices = np.argsort(dist, axis=1)
        return [_collect(ind, self.items, T)
                for ind in indices]

    def lookup(self, q):
        hashes = self.hash(q)
        return [self.get(h) for h in hashes]


class E2LSH(Index):
    def __init__(self, r, k, d):
        super(E2LSH, self).__init__(d)
        self.r = r
        self.k = k
        self.a = np.random.normal(size=(d, k))
        self.b = np.random.uniform(0, r)

    def hash(self, x):
        hashes = (x @ self.a + self.b) / self.r
        return hashes.astype(np.int32)


class SRP(Index):
    def __init__(self, k, d):
        super(SRP, self).__init__(d)
        self.k = k
        self.a = np.random.normal(size=(d, k))

    def hash(self, x):
        hashes = x @ self.a
        return np.sign(hashes)


class MultiTable():
    def __init__(self, tables):
        self.tables = tables

    def add(self, x):
        for t in self.tables:
            t.add(x)

    def lookup(self, x):
        aggregate = [set([]) for _ in self.tables]
        for t in self.tables:
            for agg, candidate in zip(aggregate, t.lookup(x)):
                agg.update(candidate)
        return aggregate
