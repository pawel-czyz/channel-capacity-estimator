import unittest
from cce.estimator import WeightedKraskovEstimator
from tests.teddybear_fabric import generate_teddybears
import numpy as np
from scipy.stats.distributions import norm


class TestWeightedMI(unittest.TestCase):
    def test_two_peaks_1(self):
        """Calculate optimal weights of teddy -> 0 and bunny -> 1. Equal sample sizes."""
        print("Running two eq amounts, uneq weights")
        prod = {"teddy": 5000, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 1/3, "bunny": 2/3}
        mi = wke.calculate_weighted_mi(weights, k=20)
        mi2 = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi, mi2, delta=0.01)

    def test_two_peaks_2(self):
        """Calculate the MI of teddy -> 0 and bunny -> 1. Unequal sample sizes."""
        print("Running two uneq amounts, eq weights")
        prod = {"teddy": 2500, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 1/2, "bunny": 1/2}
        mi = wke.calculate_weighted_mi(weights, k=5)
        self.assertAlmostEqual(mi, 1, delta=0.01)

    def test_two_peaks_3(self):
        print("Running two eq amounts, eq weights")
        prod = {"teddy": 5000, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 0.5, "bunny": 0.5}
        for k in [5, 10]:
            mi = wke.calculate_weighted_mi(weights, k=k)
            self.assertAlmostEqual(mi, 1, delta=0.01)

    def test_two_categories_of_teddies(self):
        """Calculate MI of teddy_black, teddy_brown -> 0 and bunny -> 1. Equal sample sizes and weights for each category"""
        print("Running three eq amounts and weights")
        prod = {"teddy_black": 2500, "teddy_brown": 2500, "bunny": 2500}
        fluf = {"teddy_black": 0, "teddy_brown": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy_black": 1/3, "teddy_brown": 1/3, "bunny": 1/3}
        mi = wke.calculate_weighted_mi(weights, k=8)
        mi2 = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)
        self.assertAlmostEqual(mi, mi2, delta=0.01)

if __name__ == '__main__':
    unittest.main()
