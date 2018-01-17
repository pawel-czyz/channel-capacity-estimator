import unittest
from cce.estimator import WeightedKraskovEstimator
from tests.teddybear_fabric import generate_teddybears
import numpy as np
from scipy.stats.distributions import norm


class TestWeightedMI(unittest.TestCase):
    def test_two_peaks_1(self):
        """Calculate optimal weights of teddy -> 0 and bunny -> 1. Equal sample sizes."""
        print("Running two eq amounts, uneq weights")
        prod = {"teddy": 10000, "bunny": 10000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 1/3, "bunny": 2/3}
        mi = wke.calculate_weighted_mi(weights, k=200)
        print(mi)
        #self.assertAlmostEqual(mi_i, 1, places=3)

    def test_two_peaks_2(self):
        """Calculate the MI of teddy -> 0 and bunny -> 1. Unequal sample sizes."""
        print("Running two uneq amounts, eq weights")
        prod = {"teddy": 5000, "bunny": 10000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 1/2, "bunny": 1/2}
        mi = wke.calculate_weighted_mi(weights, k=5)
        print(mi)

    def test_two_peaks_3(self):
        print("Running two eq amounts, eq weights")
        prod = {"teddy": 5000, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy": 0.5, "bunny": 0.5}
        for k in [5, 10]:
            mi = wke.calculate_weighted_mi(weights, k=k)
            print("k:", k, ", MI:", mi)

    def test_two_categories_of_teddies(self):
        """Calculate MI of teddy_black, teddy_brown -> 0 and bunny -> 1. Equal sample sizes and weights for each category"""
        print("Running three eq amounts and weights")
        prod = {"teddy_black": 5000, "teddy_brown": 5000, "bunny": 5000}
        fluf = {"teddy_black": 0, "teddy_brown": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        weights = {"teddy_black": 1/3, "teddy_brown": 1/3, "bunny": 1/3}
        mi = wke.calculate_weighted_mi(weights, k=200)
        print(mi)

if __name__ == '__main__':
    unittest.main()
