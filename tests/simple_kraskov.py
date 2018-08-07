"""This is a concise implementation of the Kraskov algorithm.
It is intended to be used for reference in unit tests only."""

from collections import Counter
from numpy import log, mean, random, abs
from scipy.spatial import cKDTree
from scipy.special import digamma

HUGE_DIST = 1e100
LEAF_SIZE = 10


def _add_noise(data):
    if not data:
        return data

    k = len(data[0][1])

    lenfactor = 1./k * (max(map(sum, abs(list(map(_project_on_ints, data))))) -\
                        min(map(sum, abs(list(map(_project_on_ints, data))))))

    def dust():
        return random.randn(k)*lenfactor*1e-6

    return [(s, (i + dust())) for s, i in data]


def _immerse(datum):
    s, il = datum
    return list([HUGE_DIST * int(i) for i in s]) + list(il)


def _project_on_ints(datum):
    return datum[1]


def _project_on_signal(datum):
    return datum[0]


def simple_calculate_mi(data, k=100):
    """Expects the data to be a list of points that look like:
       ("0101..1", [float, float, ...])
    """
    data = _add_noise(data)
    n = len(data)
    immersed_data = list(map(_immerse, data))
    only_ints = list(map(_project_on_ints, data))
    only_sigs = list(map(_project_on_signal, data))
    cnt = Counter(only_sigs)

    kdt = cKDTree(immersed_data, leafsize=LEAF_SIZE)
    kdt_ints = cKDTree(only_ints, leafsize=LEAF_SIZE)

    epses = [kdt.query(datum, k=(k+1), distance_upper_bound=HUGE_DIST)[0][-1]
             for datum in immersed_data]

    n_ints = [len(kdt_ints.query_ball_point(data[i][1], epses[i])) - 1
              for i in range(n)]

    return (digamma(k) + digamma(n) -
            mean([digamma(n_ints[i]) + digamma(cnt.get(data[i][0]))
                  for i in range(n)]))/log(2)
