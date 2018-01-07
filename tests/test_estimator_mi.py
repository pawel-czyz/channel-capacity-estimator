import unittest
from cce.estimator import WeightedKraskovEstimator
from tests.teddybear_fabric import generate_teddybears
import numpy as np


class TestEstimator(unittest.TestCase):
    def test_two_peaks_1(self):
        """Calculate the MI of teddy -> 0 and bunny -> 1. Equal sample sizes."""
        prod = {"teddy": 10000, "bunny": 10000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        mi = [wke.calculate_mi(k=k) for k in [100, 300, 1000]]
        for mi_i in mi:
            self.assertAlmostEqual(mi_i, 1, places=3)

    def test_two_peaks_2(self):
        """Calculate the MI of teddy -> 0 and bunny -> 1. Unequal sample sizes."""
        prod = {"teddy": 5000, "bunny": 10000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        mi = [wke.calculate_mi(k=k) for k in [100, 300, 1000]]
        for mi_i in mi:
            self.assertAlmostEqual(mi_i, 1, places=3)

    def test_two_categories_of_teddies(self):
        """Calculate MI of teddy_black, teddy_brown -> 0 and bunny -> 1. Equal sample sizes for each category"""
        prod = {"teddy_black": 5000, "teddy_brown": 5000, "bunny": 5000}
        fluf = {"teddy_black": 0, "teddy_brown": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        mi = [wke.calculate_mi(k=k) for k in [100, 300]]
        for mi_i in mi:
            self.assertAlmostEqual(mi_i, -(np.log(2/3)*2/3 + np.log(1/3)*1/3), places=3)

if __name__ == '__main__':
    unittest.main()
