"""Data engineering functions, as adding some dust or normalisation"""
import numpy as np

np.random.seed(98 + 105 + 111)


def _project_labels(data: list) -> list:
    return [x[0] for x in data]


def _project_coords(data: list) -> list:
    return [x[1] for x in data]


def normalise(data: list) -> list:
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


def unique(arr) -> bool:
    """Checks if array of coordinate points has all points unique.

    Parameters
    ----------
    arr : iterable
        arr [[float, float, ...], ...]

    Returns
    --------
    bool:
        True if points are distinct, False otherwise
    """
    return len(arr) == len({tuple(p) for p in arr})


def stir_norm(data: list) -> list:
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

    lab, arr = _project_labels(data), _project_coords(data)
    arr = np.array(arr)
    if not unique(arr):
        print("WARNING: data contains points that are not unique.",
              "This is known to cause numerical issues.",
              "Slightly perturbating data...")
        eps = min(filter(lambda x: x > 0,
                         {x for x in np.ndarray.flatten(arr)}))
        eps /= 2

        arr += eps * np.random.rand(*arr.shape)
        while not unique(arr):
            eps /= 2
            arr += eps * np.random.rand(*arr.shape)
            if eps == 0.:
                raise Exception("Data normalisation failed. Points cannot be stirred properly.")
        print("eps used:", eps)
        
    return list(zip(lab, arr))
