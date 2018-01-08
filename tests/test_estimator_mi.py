import unittest
from cce.estimator import WeightedKraskovEstimator
from tests.teddybear_fabric import generate_teddybears
import numpy as np
from tests.kraskov import mi_kraskov


class TestEstimator(unittest.TestCase):
    def test_two_peaks_1(self):
        """Calculate the MI of "0" -> 0 and "1" -> 1. N=2*10000, k=100"""
        prod = {"0": 10000, "1": 10000}
        fluf = {"0": 0, "1": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)

        mi1 = wke.calculate_mi(k=100)
        mi2 = mi_kraskov(data=data, k=100)

        self.assertAlmostEqual(mi1, 1, delta=0.02)
        self.assertAlmostEqual(mi1, mi2, delta=0.005)

    def test_two_peaks_2(self):
        """Calculate the MI of "0" -> 0 and "1" -> 1. N = 5000 + 10000, k=100"""
        prod = {"0": 5000, "1": 10000}
        fluf = {"0": 0, "1": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)

        mi1 = wke.calculate_mi(k=100)
        mi2 = mi_kraskov(data=data, k=100)
        mi3 = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)

        self.assertAlmostEqual(mi1, mi3, delta=0.02)
        self.assertAlmostEqual(mi1, mi2, delta=0.005)

    def test_two_peaks_3(self):
        """Calculate the MI of "0" -> 0 and "1" -> 1. N=2*1000, k=50."""
        prod = {"0": 1000, "1": 1000}
        fluf = {"0": 0, "1": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)

        mi1 = wke.calculate_mi(k=50)
        mi2 = mi_kraskov(data=data, k=50)

        self.assertAlmostEqual(mi1, 1, delta=0.03)
        self.assertAlmostEqual(mi1, mi2, delta=0.005)

    def test_doubled_category(self):
        """Calculate MI of "0", "1" -> 0 and "2" -> 1. N=3*5000, k=100"""
        prod = {"0": 5000, "1": 5000, "2": 5000}
        fluf = {"0": 0, "1": 0, "2": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)

        mi1 = wke.calculate_mi(k=100)
        mi2 = mi_kraskov(data=data, k=100)
        mi3 = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)

        self.assertAlmostEqual(mi1, mi3, delta=0.02)
        self.assertAlmostEqual(mi1, mi2, delta=0.01)

    def test_doubled_category_2(self):
        """Calculate MI of "0", "1" -> 0 and "2" -> 1. N=3*2000, k=50"""
        prod = {"0": 2000, "1": 2000, "2": 2000}
        fluf = {"0": 0, "1": 0, "2": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)

        mi1 = wke.calculate_mi(k=50)
        mi2 = mi_kraskov(data=data, k=50)
        mi3 = -(np.log2(2/3)*2/3 + np.log2(1/3)*1/3)

        self.assertAlmostEqual(mi1, mi3, delta=0.02)
        self.assertAlmostEqual(mi1, mi2, delta=0.01)


if __name__ == '__main__':
    unittest.main()
