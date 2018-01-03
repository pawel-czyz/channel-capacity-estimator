"""Weighted Kraskov estimator"""
from cce.dataeng import normalise, stir_norm, stir_unorm


class WeightedKraskovEstimator:
    _norm_huge_dist = 1e6
    _unorm_huge_dist = 1e100

    def __init__(self, k=100):
        """Weighted Kraskov Estimator

        Parameters
        ----------
        k : int
            positive integer, specifies the size of the neighborhood used in the estimation

        """

        # Set k
        self._k = None
        self.k = k

        # Define huge distance
        self._huge_dist = None

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if type(value) != int:
            raise TypeError("k must be int")

        if value < 1:
            raise ValueError("k must be positive")

        self._k = value

    def _immerse(self, data):
        """label-value space is immersed in Euclidean space

        Parameters
        ----------
        data : list
            a list of tuples (label, value)

        Returns
        -------
        data : list
            list of lists representing points in Euclidean space. Points with different labels belong to different
            parallel planes, separated by huge distance.
        """
        if self._huge_dist is None:
            raise Exception("Huge distance not defined.")

        lab_vals = dict()
        immersed = []
        sep_coor = 0

        # Sort data for better branch prediction.
        data = sorted(data, key=hash)

        for label, value in data:
            if label not in lab_vals:
                sep_coor += self._huge_dist
                lab_vals[label] = sep_coor

            immersed.append([lab_vals[label]] + list(value))

        return immersed

    def load(self, data, normalisation=True):
        """Loads data to the tree.

        Parameters
        ----------
        data : list
            data is a list of tuples. Each tuples has form (label, value), where label can be either int or string
            and value is a one-dimensional numpy array/list representing coordinates
        normalisation : bool
            flag that decided if the data should be normalised before being processed further
        """

        # Decide whether the data should be normalised and choose the stirring method
        if normalisation:
            self._huge_dist = self._norm_huge_dist
            data = normalise(data)
            data = stir_norm(data)
        else:
            self._huge_dist = self._unorm_huge_dist
            data = stir_unorm(data)

        immersed_data = self._immerse(data)

