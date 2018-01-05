"""Weighted Kraskov estimator"""
from cce.dataeng import normalise, stir_norm, cut_first_coordinate
from cce.optimize_weights import weight_optimizer
from scipy.spatial import cKDTree
from scipy.special import digamma
from collections import defaultdict
import numpy as np


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

        # Save in a defaultdict total number of points for a given label
        self._number_of_points = defaultdict(lambda: 0)
        self._number_of_labels = None

        # TODO - write getter and setter
        # Trees with points
        self.tree_full = None
        self.tree_coordinates = None

        # Immersed data
        self._immersed_data_full = None
        self._immersed_data_coordinates = None

        # Array of labels and array of neighborhood
        self._array_labels = None
        self._array_neighs = None

        # If the data are readable
        self._data_prepared = False



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

            # Add this point to the summary how many points are for each label_index
            self._number_of_points[label_index] += 1

        # Shuffle the data for better k-d tree performance
        np.random.shuffle(immersed_data)

        # Create: data, just coordinates, tree on the data, tree on the coordinates
        self._immersed_data_full = np.array(immersed_data)
        self._immersed_data_coordinates = immersed_data[:, 1:]
        self.tree_full = cKDTree(self._immersed_data_full, leafsize=self.leaf_size)
        self.tree_coordinates = cKDTree(self._immersed_data_coordinates, leafsize=self.leaf_size)
        self._number_of_labels = index

        # Calculate array of labels
        self._array_labels = immersed_data[:, 0]

        # Toggle the flag that everything is ready to be read
        self._data_prepared = True

    def calculate_mi(self):
        """Calculates MI using Kraskov estimation on the previously loaded data.

        Returns
        -------
        float
            mutual information in bits
        """

        if not self._data_prepared:
            raise Exception("Data have not been loaded yet.")

        epses = [self.tree_full.query(datum, k=(self.k + 1), distance_upper_bound=self._huge_dist)[0][-1]
                 for datum in self._immersed_data_full]

        n = len(self._immersed_data_full)

        n_y = [len(self.tree_coordinates.query_ball_point(self._immersed_data_coordinates[i], epses[i])) - 1
               for i in range(n)]

        return (
               digamma(self.k) + digamma(n) -
               np.mean([digamma(n_y[i])+digamma(self._number_of_points[self._immersed_data_full[i][0]])
                        for i in range(n)])
           ) / np.log(2)

    def neighborhood_array(self):
        return self._array_neighs

    def labels_array(self):
        return self._array_labels

    def optimise_weights(self):
        return weight_optimizer(neighb_count=self.neighborhood_array(), labels=self.labels_array())

    def _make_into_neigh_list(self, indices, special_point_label):
        labels = self._array_labels[indices]
        neigh_list = np.zeros(self._number_of_labels)
        for lab in labels:
            neigh_list[lab] += 1

        neigh_list[special_point_label] -= 1

        return neigh_list

    def calculate_neighborhood(self):
        if not self._data_prepared:
            raise Exception("Data have not been loaded yet.")

        epses = [self.tree_full.query(datum, k=(self.k + 1), distance_upper_bound=self._huge_dist)[0][-1]
                 for datum in self._immersed_data_full]

        neighs = [
            self._make_into_neigh_list(
                self.tree_coordinates.query_ball_point(coord, epses[i]),
                self._array_labels[i])
            for i, coord in enumerate(self._immersed_data_coordinates)]

        self._array_neighs = np.array(neighs)
