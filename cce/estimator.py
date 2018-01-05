"""Weighted Kraskov estimator"""
from cce.dataeng import normalise, stir_norm, cut_first_coordinate
from collections import Counter
from scipy.spatial import cKDTree
import numpy as np
from copy import deepcopy

np.random.seed(154)


class WeightedKraskovEstimator:
    def __init__(self, k=100, leaf_size = None):
        """Weighted Kraskov Estimator

        Parameters
        ----------
        k : int
            positive integer, specifies the size of the neighborhood used in the estimation

        """

        # Set k
        self._k = None
        self.k = k

        # Set leaf size
        self._leaf_size = None
        if leaf_size is not None:
            self.leaf_size = leaf_size
        else:
            self.leaf_size = 16

        # Define huge distance
        self._huge_dist = 1e6

        # Define dictionaries converting between labels and numpy array indices
        self._label2index = dict()
        self._index2label = dict()

        # TODO - write getter and setter
        # Trees with points
        self.full_tree = None
        self.coordinate_tree = None

        # Immersed data
        self._immersed_data = None

    @property
    def leaf_size(self):
        return self._leaf_size

    @leaf_size.setter
    def leaf_size(self, value):
        if type(value) != int:
            raise TypeError("leaf_size must be int")
        if value < 1:
            raise ValueError("leaf_size must be positive")
        self._leaf_size = value

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

    def load(self, data):
        """Loads data into the structure.

        Parameters
        ----------
        data : list
            data is a list of tuples. Each tuples has form (label, value), where label can be either int or string
            and value is a one-dimensional numpy array/list representing coordinates
        """

        # Normalise the data
        normalised_data = normalise(data)

        # Add small peturbation to the data
        normalised_data = stir_norm(normalised_data)

        # Sort data for better branch prediction.
        normalised_data = sorted(normalised_data, key=hash)
        # `immersed_data` is a list of Euclidean points
        immersed_data = []
        # `index` is a unique int for every label. It will have values 0, 1, 2, ...
        index = 0

        for label, value in normalised_data:
            # If it is the first time we see this label, we create a new category for it.
            if label not in self._label2index:
                self._label2index[label] = index
                self._index2label[index] = label
                index += 1

            # Now we are sure that considered label has it's unique index.
            label_index = self._label2index[label]
            separating_coordinate = label_index * self._huge_dist
            immersed_data.append([separating_coordinate] + list(value))

        # Shuffle the data for better k-d tree performance
        np.random.shuffle(immersed_data)

        self.full_tree = cKDTree(immersed_data, leafsize=self.leaf_size)
        self.coordinate_tree = cKDTree(cut_first_coordinate(immersed_data), leafsize=self.leaf_size)
        self._immersed_data = immersed_data

    def calculate_mi(self):
        """Calculates MI using Kraskov estimation ond built trees

        Returns
        -------
        float
            mutual information
        """

        if self.full_tree is None or self.coordinate_tree is None:
            raise Exception("Data have not been loaded yet.")


        epses = [kdt.query(datum, k=(k+1), distance_upper_bound=_huge_dist)[0][-1]
                    for datum in immersed_data]

    n_ints = [len(kdt_ints.query_ball_point(data[i][1], epses[i])) - 1
              for i in range(n)]

    return (
               digamma(k) + digamma(n) - \
               mean(
                   [digamma(n_ints[i])+digamma(cnt.get(data[i][0])) for i in range(n)]
               )
           ) / log(2)

