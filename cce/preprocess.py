# This file is part of Channel Capacity Estimator,
# licenced under GNU GPL 3 (see file License.txt).
# Homepage: http://pmbm.ippt.pan.pl/software/cce 

"""Input data preprocessing functions"""

import numpy as np


def _project_labels(data: list) -> list:
    return [x[0] for x in data]


def _project_coords(data: list) -> list:
    return [x[1] for x in data]


def normalize(data: list) -> list:
    """Perform input data normalization

    Parameters
    ----------
    data : list
        data is a list of tuples. Each tuple has form (label, value),
        where label can be either int or string and value is a
        one-dimensional numpy array/list representing coordinates.

    Returns
    -------
    list
        list of data points. Each data point is normalized to interval
        [0, 1].
    """
    lab = _project_labels(data)
    arr = _project_coords(data)
    arr = np.array(arr)

    max_ = np.amax(arr)
    min_ = np.amin(arr)

    arr = (arr - min_)/(max_ - min_)

    return list(zip(lab, arr))


def unique(arr) -> bool:
    """Check if all points in the array of coordinates are unique.

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


def add_noise_if_duplicates(data: list) -> list:
    """Add noise to input data

    Parameters
    ----------
    data : list
        data [(label, value), ...], where value is list of floats

    Returns
    -------
    list
        data with added noise
    """

    lab, arr = _project_labels(data), _project_coords(data)
    arr = np.array(arr)
    if not unique(arr):
        print("WARNING: data contains points that are not unique.",
              "Data will be perturbed to avoid numerical issues.")
        eps = min(filter(lambda x: x > 0,
                         {x for x in np.ndarray.flatten(arr)}))
        eps /= 2

        arr += eps * np.random.rand(*arr.shape)
        while not unique(arr):
            eps /= 2
            arr += eps * np.random.rand(*arr.shape)
            if eps == 0.:
                raise Exception("Cannot add noise to input data.")
        print("eps used:", eps)

    return list(zip(lab, arr))
