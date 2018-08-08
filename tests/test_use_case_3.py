import unittest
import numpy as np
from scipy.stats.distributions import norm
from cce.estimator import WeightedKraskovEstimator as wke
from tests.noisy_channel import communicate

NN_K = 30
DELTA = 0.01

class TestUseCase3(unittest.TestCase):
    """Test of maximum mutual information (max MI, capacity) estimation 
    with optimization of weights of input distributions."""

    def test_equal_sizes(self):
        data = communicate(input_counts={'A': 5000, 'B': 5000},
                           output_values={'A': 0.0, 'B': 1.0})
        capacity_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        capacity_exact = 1.0
        self.assertAlmostEqual(capacity_est, capacity_exact, delta=DELTA)
        self.assertAlmostEqual(opt_ws['A'], 0.5, delta=DELTA)
        self.assertAlmostEqual(opt_ws['B'], 0.5, delta=DELTA)

    def test_unequal_sizes(self):
        data = communicate({'A': 2500, 'B': 5000},
                           {'A': 0.0, 'B': 1.0})
        capacity_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        capacity_exact = 1.0
        self.assertAlmostEqual(capacity_est, capacity_exact, delta=DELTA)
        self.assertAlmostEqual(opt_ws['A'], 0.5, delta=DELTA)
        self.assertAlmostEqual(opt_ws['B'], 0.5, delta=DELTA)

    def test_equal_sizes_partial_overlap(self):
        data = communicate({'A': 2500, 'B': 2500, 'C': 2500},
                           {'A': 0, 'B': 0, 'C': 1})
        capacity_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        capacity_exact = 1.0
        self.assertAlmostEqual(capacity_est, capacity_exact, delta=0.01)
        self.assertAlmostEqual(opt_ws['C'], 0.5, delta=DELTA)

    def test_eight_gaussians(self):
        """This test case is patterned after a test included in the source code 
        featuring an article by Tudelska et al. (Scientific Reports, 2017), 
        DOI: doi.org/10.1038/s41598-017-16166-y ."""

        def _mu_sigma(i):
            return (1.5*(i + 1), (i + 1)/4. + 1)

        data = []
        for dist_i, count in enumerate(8*[2500]):
            data.extend([(dist_i, [values]) 
                         for values in norm(*_mu_sigma(dist_i)).rvs(count)])

        capacity_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        largest_2_opt_ws_indices = list(map(lambda kv: kv[0], 
                sorted(opt_ws.items(), key=lambda kv: kv[1], reverse=True)[0:2]))
        capacity_accurate, largest_2_opt_ws_indices_accurate = 1.14425, [0,7]
        self.assertAlmostEqual(capacity_est, capacity_accurate, delta=DELTA)
        self.assertEqual(largest_2_opt_ws_indices, largest_2_opt_ws_indices_accurate)


if __name__ == '__main__':
    unittest.main()
