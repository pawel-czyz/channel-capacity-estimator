import unittest
from cce.estimator import WeightedKraskovEstimator
from tests.teddybear_fabric import generate_teddybears
import numpy as np
from scipy.stats.distributions import norm


class TestWeightOptimizer(unittest.TestCase):
    def test_two_peaks_1(self):
        """Calculate optimal weights of teddy -> 0 and bunny -> 1. Equal sample sizes."""
        prod = {"teddy": 5000, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        wke.calculate_neighborhood(k=10)
        mi, ws = wke.optimize_weights()
        self.assertAlmostEqual(mi, 1, delta=0.01)
        self.assertAlmostEqual(ws["teddy"], 0.50, delta=0.01)
        self.assertAlmostEqual(ws["bunny"], 0.50, delta=0.01)

    def test_two_peaks_2(self):
        """Calculate the MI of teddy -> 0 and bunny -> 1. Unequal sample sizes."""
        prod = {"teddy": 2500, "bunny": 5000}
        fluf = {"teddy": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        wke.calculate_neighborhood(k=10)
        mi, ws = wke.optimize_weights()
        self.assertAlmostEqual(mi, 1, delta=0.01)
        self.assertAlmostEqual(ws["teddy"], 0.50, delta=0.01)
        self.assertAlmostEqual(ws["bunny"], 0.50, delta=0.01)

    def test_two_categories_of_teddies(self):
        """Calculate MI of teddy_black, teddy_brown -> 0 and bunny -> 1. Equal sample sizes for each category"""
        prod = {"teddy_black": 2500, "teddy_brown": 2500, "bunny": 2500}
        fluf = {"teddy_black": 0, "teddy_brown": 0, "bunny": 1}
        data = generate_teddybears(prod, fluf, 0.0001)
        wke = WeightedKraskovEstimator(data)
        wke.calculate_neighborhood(k=10)
        mi, ws = wke.optimize_weights()
        self.assertAlmostEqual(mi, 1, delta=0.01)
        self.assertAlmostEqual(ws["bunny"], 0.50, delta=0.01)

#    def test_gaussians(self):
#        print("Calculate optimal MI for a couple Gaussians")
#        def _mu_sigma(i):
#            return (1.5*(i+1), (i+1)/4. + 1)
#
#        subsamples_counts = 10 * np.array([257, 563, 533, 440, 537, 625, 503,
#            747])
#
#        data = []
#        for i in range(len(subsamples_counts)):
#            for v in norm(*_mu_sigma(i)).rvs(subsamples_counts[i]):
#                data.append((i, [v]))
#
#        wke = WeightedKraskovEstimator(data)
#        wke.calculate_neighborhood(k=200)
#        mi, ws = wke.optimise_weights()
#        print(mi, ws)

if __name__ == '__main__':
    unittest.main()
