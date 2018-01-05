"""Data engineering functions, as adding some dust or normalisation"""
import numpy as np

np.random.seed(98 + 105 + 111)


def _project_labels(data):
    return [x[0] for x in data]


def _project_coords(data):
    return [x[1] for x in data]


def normalise(data):
    """Data normalisation

    Parameters
    ----------
    data : list
        data is a list of tuples. Each tuples has form (label, value), where label can be either int or string
        and value is a one-dimensional numpy array/list representing coordinates

    Returns
    -------
    list
        list of data points. Each data point is normalised, what means that each coordinate is in the interval
        [0, 1]
    """
    lab = _project_labels(data)
    arr = _project_coords(data)
    arr = np.array(arr)

    b = np.amax(arr)
    a = np.amin(arr)

    arr = (arr-a) / (b-a)

    return list(zip(lab, arr))


def stir_norm(data):
    """Stirs normalised data

    Parameters
    ----------
    data : list
        data [(label, value), ...], where value is list of floats

    Returns
    -------
    list
        data but with added noise
    """
    magnitude = 1e-8

    lab, arr = _project_labels(data), _project_coords(data)
    arr = np.array(arr)
    arr += magnitude * np.random.rand(*arr.shape)
    return list(zip(lab, arr))


def stir_unorm(data):
    """Stirs unnormalised data. Warning: this function is deprecated.

    Parameters
    ----------
    data : list
        data [(label, value), ...], where value is list of floats

    Returns
    -------
    list
        data but with added noise
    """
    if len(data) == 0:
        return data

    k = len(data[0][1])

    coord_sizes = np.sum(np.abs(_project_coords(data)), axis=1)

    lenfactor = (coord_sizes.max() - coord_sizes.min()) / k

    def dust():
        return np.random.randn(k) * lenfactor * 1e-6

    return [(s, i + dust()) for s, i in data]


def cut_first_coordinate(arr):
    """For 2d list of Euclidean points cuts the first coordinate

    Parameters
    ----------
    arr : ndarray
        shape (number of points, dimension). Alternatively it can be a list of points

    Returns
    -------
    ndarray
        shape (number of points, dimension-1) - array with first (0th) coordinate cut
    """

    npy_arr = np.array(arr)
    return npy_arr[:, 1:]