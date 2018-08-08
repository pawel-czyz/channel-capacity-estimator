import unittest
import numpy as np
from cce.estimator import WeightedKraskovEstimator as wke
from tests.noisy_channel import communicate

NN_K = 30
DELTA = 0.01

class TestUseCase2(unittest.TestCase):
    """Tests of mutual information (MI) estimation for input distributions
    with imposed weights."""

    def test_equal_sizes(self):
        data = communicate(input_counts={'A': 5000, 'B': 5000},
                           output_values={'A': 0.0, 'B': 1.0})
        weights = {'A': 1/3, 'B': 2/3}
        mi_est = wke(data).calculate_weighted_mi(weights, k=NN_K)
        mi_exact = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi_est, mi_exact, delta=DELTA)

    def test_unequal_sizes(self):
        data = communicate({'A': 2500, 'B': 5000},
                           {'A': 0.0, 'B': 1.0})
        weights = {'A': 1/2, 'B': 1/2}
        mi_est = wke(data).calculate_weighted_mi(weights, k=NN_K)
        mi_exact = 1.0
        self.assertAlmostEqual(mi_est, mi_exact, delta=DELTA)

    def test_unequal_sizes_trivial_overlap(self):
        data = communicate({'A': 2500, 'B': 5000},
                           {'A': 0.0, 'B': 0.0})
        weights = {'A': 1/2, 'B': 1/2}
        mi_est = wke(data).calculate_weighted_mi(weights, k=NN_K)
        mi_exact = 0.0
        self.assertAlmostEqual(mi_est, mi_exact, delta=DELTA)

    def test_output_overlap(self):
        data = communicate({'A': 2500, 'B': 2500, 'C': 2500},
                           {'A': 0, 'B': 0, 'C': 1})
        weights = {'A': 1/3, 'B': 1/3, 'C': 1/3}
        mi_est = wke(data).calculate_weighted_mi(weights, k=NN_K)
        mi_exact = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi_est, mi_exact, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
