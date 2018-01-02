"""Weighted Kraskov estimator"""


class WeightedKraskovEstimator:
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

