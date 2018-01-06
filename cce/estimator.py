"""Weighted Kraskov estimator"""
from cce.dataeng import normalise, stir_norm, cut_first_coordinate
from cce.optimize_weights import weight_optimizer
from scipy.spatial import cKDTree
from scipy.special import digamma
from collections import defaultdict
import numpy as np


np.random.seed(154)


class WeightedKraskovEstimator:
    # Constant separating points with different labels (X)
    _huge_dist = 1e6

    def __init__(self, leaf_size=16):
        """Weighted Kraskov Estimator

        Parameters
        ----------
        leaf_size : int
            positive integer, used for tree construction
        """

        # Set leaf size
        self.leaf_size = leaf_size

        # Define dictionaries converting between labels and numpy array indices
        self._label2index = dict()
        self._index2label = dict()

        # Save in a defaultdict total number of points for a given label
        self._number_of_points_for_label = defaultdict(lambda: 0)
        self._number_of_labels = None

        # Immersed data - X is mapped from abstract object into reals using _huge_dist and we store spaces
        # X x Y and just Y
        self._immersed_data_full = None
        self._immersed_data_coordinates = None
        # Total number of points
        self._number_of_points_total = None

        # Trees with points - tree_full stores X x Y and tree_coordinates stores just Y
        self.tree_full = None
        self.tree_coordinates = None

        # Array of labels and array of neighborhood for each point
        self.label_array = None
        self.neighborhood_array = None

        # If the data are readable
        self._data_loaded = False

        # We store last used k and calculated epsilon for each point. It theoretically speeds up the algorithm
        # but costs additional memory
        self._k = None
        self._epsilons = None
        # If _new_data_loaded is True, we know that we need to recalculate epsilons
        self._new_data_loaded = False

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
        # `immersed_data` is a list of Euclidean points: [ [label_real, coordinate_1, coordinate_2, ...], ... ]
        immersed_data = []
        # `index` is a unique int for every label. It will have values 0, 1, 2, ...
        index = 0

        for label, value in normalised_data:
            # If it is the first time we see this label, we create a new category for it.
            if label not in self._label2index:
                self._label2index[label] = index
                self._index2label[index] = label
                index += 1

            # Now we are sure that considered label has unique index.
            label_index = self._label2index[label]
            separating_coordinate = label_index * self._huge_dist
            immersed_data.append([separating_coordinate] + list(value))

            # Add this point to the summary how many points are for each label_index
            self._number_of_points_for_label[label_index] += 1

        # Number of different labels (X) is just index now
        self._number_of_labels = index

        # Shuffle the data for better k-d tree performance
        np.random.shuffle(immersed_data)

        # Create immersed data and data without label information (X x Y and Y)
        self._immersed_data_full = np.array(immersed_data)
        self._immersed_data_coordinates = immersed_data[:, 1:]

        # Calculate array of labels
        self.label_array = immersed_data[:, 0]

        # Calculate the number of points
        self._number_of_points_total = len(self._immersed_data_full)

        # Put the data into the k-d trees
        self.tree_full = cKDTree(self._immersed_data_full, leafsize=self.leaf_size)
        self.tree_coordinates = cKDTree(self._immersed_data_coordinates, leafsize=self.leaf_size)

        # Toggle the flag that the trees are ready to use
        self._data_loaded = True
        # Toggle the flag that new data appeared and we need to recalculate epsilons
        self._new_data_loaded = True

    def _prepare_epsilons(self, k):
        """Returns the list of epsilons. Uses caching.

        Parameters
        ----------
        k : int
            number of points in neighborhood used for new estimation

        Returns
        -------
        list
            list of floats, one for each point
        """
        # If in the meantime the data have changed or someone wants to use different k, we need to recalculate
        # epsilons. Otherwise we can use cached value
        if self._new_data_loaded or k != self._k:
            self._k = k
            self._epsilons = [self.tree_full.query(datum, k=k+1, distance_upper_bound=self._huge_dist)[0][-1]
                              for datum in self._immersed_data_full]
            self._new_data_loaded = False
        return self._epsilons

    def _check_if_data_are_loaded(self):
        if not self._data_loaded:
            raise Exception("Data have not been loaded yet.")

    def calculate_mi(self, k=100):
        """Calculates MI using Kraskov estimation on the previously loaded data.

        Parameters
        ----------
        k : int
            how many neighboring points should be used in the estimation

        Returns
        -------
        float
            mutual information in bits
        """
        self._check_if_data_are_loaded()

        n = self._number_of_points_total
        epses = self._prepare_epsilons(k=k)

        n_y = [len(self.tree_coordinates.query_ball_point(self._immersed_data_coordinates[i], epses[i])) - 1
               for i in range(n)]

        def _n_x_for_a_given_point_index(i):
            return self._number_of_points_for_label[self.label_array[i]]

        return (
               digamma(k) + digamma(n) -
               np.mean([digamma(n_y[i])+digamma(_n_x_for_a_given_point_index(i))
                        for i in range(n)])
           ) / np.log(2)

    def optimise_weights(self):
        """Function optimising weights using weight_optimizer - the output is still under consideration."""
        if self._new_data_loaded:
            raise Exception("New data have been loaded.")
        return weight_optimizer(neighb_count=self.neighborhood_array, labels=self.label_array)

    def _make_into_neigh_list(self, indices, special_point_label):
        """Prepare a column (row?) of neighborhood matrix.

        Parameters
        ----------
        indices : list
            the list of indices of the points in the neighborhood, as given from tree
        special_point_label : int
            label for the point for which the neighborhood is calculated, as we need to subtract this point

        Returns
        -------
        ndarray
            calculated neighborhoods
        """
        labels = self.label_array[indices]
        neigh_list = np.zeros(self._number_of_labels)
        for lab in labels:
            neigh_list[lab] += 1

        neigh_list[special_point_label] -= 1

        return neigh_list

    def calculate_neighborhood(self, k):
        """Function that prepared neighborhood_array. It may be still changed, as we may decideto cache more data"""
        self._check_if_data_are_loaded()
        epses = self._prepare_epsilons(k=k)

        neighs = [
            self._make_into_neigh_list(
                self.tree_coordinates.query_ball_point(coord, epses[i]),
                self.label_array[i])
            for i, coord in enumerate(self._immersed_data_coordinates)]

        self.neighborhood_array = np.array(neighs)
