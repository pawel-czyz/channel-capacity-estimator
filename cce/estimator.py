# This file is part of Channel Capacity Estimator,
# licenced under GNU GPL 3 (see file License.txt).
# Homepage: http://pmbm.ippt.pan.pl/software/cce 

"""Weighted Kraskov estimator"""

from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.special import digamma
import numpy as np
from cce.preprocess import normalize, add_noise_if_duplicates
from cce.optimize import weight_optimizer
from cce.score import weight_loss

# default number of neighbors
_DEF_K = 100


class WeightedKraskovEstimator:

    # constant separating points with different labels (X)
    _huge_dist = 1e6

    def __init__(self, data: list = None, leaf_size: int = 16):
        """Weighted Kraskov Estimator

        Parameters
        ----------
        data : list
            check `load` method
        leaf_size : int
            positive integer, used for tree construction
        """
        self.leaf_size = leaf_size

        # Define dictionaries bidirectionally mapping labels and numpy 
        # array indices.
        self._label2index = dict()
        self._index2label = dict()

        # Save in a defaultdict total number of points for a given label.
        self._number_of_points_for_label = defaultdict(lambda: 0)
        self._number_of_labels = None

        # Immersed data -- X is mapped from cateogorical data into reals
        # using _huge_dist, and we store spaces X x Y and just Y.
        self._immersed_data_full = None
        self._immersed_data_coordinates = None

        self._number_of_points_total = None

        # Trees with points - tree_full stores X x Y, whereas 
        # tree_coordinates stores just Y.
        self.tree_full = None
        self.tree_coordinates = None

        # Array of labels and array of neighborhood for each point.
        self.label_array = None
        self.neighborhood_array = None

        self._data_loaded = False

        # We store last used k, to enable restoring the arrays from cache
        # (when the data and k have not changed).
        self._k = None

        # If _new_data_loaded is True, we know that we have to recalculate
        # epsilons.
        self._new_data_loaded = False

        # Data may be provided during the initialization. We have to load 
        # them at the end of __init__().
        if data is not None:
            self.load(data)


    def calculate_maximized_mi(self, k: int = _DEF_K):
        """High-level function calculating maximal MI.

        Parameters
        ----------
        k : int
            free parameter in Kraskov estimator

        Returns
        -------
        float
            optimized MI, in bits
        dict
            dictionary mapping labels to weights
        """
        self.calculate_neighborhoods(k=k)
        return self.optimize_weights()


    def load(self, data: list):
        """Loads data into the structure.

        Parameters
        ----------
        data : list
            data is a list of tuples, each of the form (label, value),
            where label can be either int or string and value is a
            one-dimensional numpy array/list representing coordinates
        """
        normalized_data = add_noise_if_duplicates(normalize(data))

        # Sort data for better branch prediction.
        # normalized_data = sorted(normalized_data, key=hash)
        # `immersed_data` is a list of Euclidean points:
        # [ [label_real, coordinate_1, coordinate_2, ...], ... ]
        immersed_data = []
        # `index` is a unique int for every label. It will have values
        # 0, 1, 2, ...
        index = 0
        # We want to build array of labels in this step
        label_array_tmp = []

        for label, value in normalized_data:
            # If we see this label for the first time, we create a new 
            # category for it.
            if label not in self._label2index:
                self._label2index[label] = index
                self._index2label[index] = label
                index += 1

            # Now we are sure that the considered label has unique index.
            label_index = self._label2index[label]
            separating_coordinate = label_index * self._huge_dist
            immersed_data.append([separating_coordinate] + list(value))
            label_array_tmp.append(label_index)

            # Add this point to the summary how many points are for each 
            # label_index.
            self._number_of_points_for_label[label_index] += 1

        # Number of different labels (X) is just index now.
        self._number_of_labels = index

        # Create immersed data and data without label information (X x Y and Y).
        self._immersed_data_full = np.array(immersed_data)
        self._immersed_data_coordinates = self._immersed_data_full[:, 1:]

        # Calculate array of labels.
        self.label_array = np.array(label_array_tmp, dtype=np.uint32)

        # Calculate the number of points.
        self._number_of_points_total = len(self._immersed_data_full)

        # Put the data into the k-d trees.
        self.tree_full = cKDTree(self._immersed_data_full,
                                 leafsize=self.leaf_size)
        self.tree_coordinates = cKDTree(self._immersed_data_coordinates,
                                        leafsize=self.leaf_size)

        # Toggle the flag that the trees are ready to use.
        self._data_loaded = True

        # Toggle the flag that new data appeared and we need to recalculate
        # epsilons.
        self._new_data_loaded = True


    def _check_if_data_are_loaded(self):
        if not self._data_loaded:
            raise Exception("Data have not been loaded yet.")


    def calculate_mi(self, k: int = _DEF_K) -> float:
        """Calculates MI using Kraskov estimation using previously loaded data.

        Parameters
        ----------
        k : int
            how many neighboring points should be used in the estimation

        Returns
        -------
        float
            mutual information in bits
        """
        n = self._number_of_points_total
        self.calculate_neighborhoods(k=k)

        # Calculate the number of points in neighborhood.
        n_y = self.neighborhood_array.sum(axis=1)

        def _n_x_for_a_given_point_index(i):
            return self._number_of_points_for_label[self.label_array[i]]

        n_x = [_n_x_for_a_given_point_index(i) for i in range(n)]
        digammas = digamma(n_y) + digamma(n_x)

        return (digamma(k) + digamma(n) - digammas.mean()) / np.log(2)


    def calculate_weighted_mi(self, weights: dict, k: int = _DEF_K) -> float:
        """Calculates mutual information for weighted input in bits.

        Parameters
        ----------
        weights : dict
            dictionary {label1: weight1, label2: weight2, ...}. All
            weights should sum up to 1 and be non-negative floats
        k : int
            positive int indicating the size of the neighborhood

        Returns
        -------
        float
            mutual information in bits
        """
        self.calculate_neighborhoods(k=k)

        w_list = [weights[self._index2label[i]] for i in range(self._number_of_labels)]

        # Check whether weights sum up to (almost) 1
        if abs(sum(w_list) - 1) > 0.01:
            raise ValueError("Weights should sum up to 1.")

        loss = weight_loss(neighb_count=self.neighborhood_array,
                           labels=self.label_array, weights=w_list)

        k = self._k
        n = self._number_of_points_total
        optimized_mi = digamma(k) + digamma(n) - loss

        return optimized_mi / np.log(2)


    def optimize_weights(self) -> tuple:
        """Function optimizing weights using weight_optimizer.

        Returns
        -------
        float
            optimized MI, in bits
        dict
            dictionary mapping labels to weights
        """
        if self._new_data_loaded:
            raise Exception("New data have been loaded. You need to invoke calculate_neighborhoods().")

        # Get loss and best weights from TensorFlow.
        loss, w = weight_optimizer(neighb_count=self.neighborhood_array,
                                   labels=self.label_array)

        # Get back initial labels.
        w_dict = {self._index2label[i]: w for i, w in enumerate(w)}

        # Just a final touch :)
        k = self._k
        n = self._number_of_points_total
        optimized_mi = digamma(k) + digamma(n) - loss

        return optimized_mi / np.log(2), w_dict


    def _turn_into_neigh_list(self, indices, special_point_label):
        """Prepares a row of neighborhood matrix.

        Parameters
        ----------
        indices : list
            the list of indices of the points in the neighborhood, as given from tree
        special_point_label : int
            label for the point for which the neighborhood is calculated,
            as we need to subtract this point

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


    def calculate_neighborhoods(self, k: int = _DEF_K):
        """Function that prepares neighborhood_array.

        Parameters
        ----------
        k : int
            free parameter in Kraskov estimator
        """
        self._check_if_data_are_loaded()

        # If the data nor neighborhood specification haven't changed,
        # we can use cached value...
        if not self._new_data_loaded and k == self._k:
            return

        # ...otherwise we need to recalculate everything.
        self._k = k
        epses = [self.tree_full.query(datum, k=k+1,
                                      distance_upper_bound=self._huge_dist)[0][-1]
                 for datum in self._immersed_data_full]

        neighs = [
            self._turn_into_neigh_list(
                self.tree_coordinates.query_ball_point(coord, epses[i]),  # - 1e-10),
                self.label_array[i])
            for i, coord in enumerate(self._immersed_data_coordinates)]

        self.neighborhood_array = np.array(neighs)

        # Turn off the flag with fresh data.
        self._new_data_loaded = False
