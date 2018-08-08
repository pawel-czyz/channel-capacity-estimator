import unittest
import numpy as np
from cce.estimator import WeightedKraskovEstimator as wke
from tests.plain_kraskov import calculate_mi as plain_calculate_mi
from tests.noisy_channel import communicate

NN_K = 30
DELTA = 0.01

class TestUseCase1(unittest.TestCase):
    """Tests of mutual information (MI) estimation for input distributions
    without imposed weights."""

    def test_equal_sizes_trivial_separate(self):
        data = communicate(input_counts={'0': 10000, '1': 10000}, 
                           output_values={'0': 0.0, '1': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = 1.0
        self.assertAlmostEqual(mi_est1, mi_exact, delta=DELTA)
        self.assertAlmostEqual(mi_est1, mi_est2, delta=DELTA)

    def test_equal_sizes_trivial_overlap(self):
        data = communicate({'0': 10000, '1': 10000},
                           {'0': 0.0, '1': 0.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = 0.0
        self.assertAlmostEqual(mi_est1, mi_exact, delta=DELTA)
        self.assertAlmostEqual(mi_est1, mi_est2, delta=DELTA)

    def test_unequal_sizes(self):
        data = communicate({'0': 5000, '1': 10000},
                           {'0': 0.0, '1': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi_est1, mi_exact, delta=DELTA)
        self.assertAlmostEqual(mi_est1, mi_est2, delta=DELTA)

    def test_equal_sizes_partial_overlap(self):
        data = communicate({'0': 5000, '1': 5000, '2': 5000},
                           {'0': 0.0, '1': 0.0, '2': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi_est1, mi_exact, delta=DELTA)
        self.assertAlmostEqual(mi_est1, mi_est2, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
