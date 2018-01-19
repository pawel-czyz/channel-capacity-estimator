from numpy import log, mean, random, abs
from scipy.spatial import cKDTree
from scipy.special import digamma
from collections import Counter

_huge_dist = 1e100
_leaf_size = 10


def _stir(data):
    if len(data) == 0:
        return data

    k = len(data[0][1])

    lenfactor = 1./k * (max(map(sum, abs(list(map(_project_on_ints, data))))) -\
                        min(map(sum, abs(list(map(_project_on_ints, data))))))

    def dust():
        return random.randn(k)*lenfactor*1e-6

    return [(s, (i + dust())) for s, i in data]


def _immerse(datum):
    s, il = datum
    return list([_huge_dist * int(i) for i in s]) + list(il)


def _project_on_ints(datum):
    return datum[1]


def _project_on_signal(datum):
    return datum[0]


def mi_kraskov(data, k=100):
    """Expects the data to be a list of points that look like:
       ("0101..1", [float, float, ...])
    """
    data = _stir(data)
    n = len(data)
    immersed_data = list(map(_immerse, data))
    only_ints = list(map(_project_on_ints, data))
    only_sigs = list(map(_project_on_signal, data))
    cnt = Counter(only_sigs)

    kdt = cKDTree(immersed_data, leafsize=_leaf_size)
    kdt_ints = cKDTree(only_ints, leafsize=_leaf_size)

    epses = [kdt.query(datum, k=(k+1), distance_upper_bound=_huge_dist)[0][-1]
                for datum in immersed_data]

    n_ints = [len(kdt_ints.query_ball_point(data[i][1], epses[i] - 1e-10))
                for i in range(n)]

    return (
        digamma(k) + digamma(n) -\
        mean(
            [digamma(n_ints[i])+digamma(cnt.get(data[i][0])) for i in range(n)]
        )
    ) / log(2)
