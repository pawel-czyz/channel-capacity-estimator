import unittest
from itertools import product
from numpy import cos, sin, log2, identity
from scipy.stats import multivariate_normal as multinorm
from scipy.stats.distributions import norm
from cce.estimator import WeightedKraskovEstimator as wke
from tests.noisy_channel import communicate

NN_K = 15
ATOL = 0.02
RTOL = 0.04

class TestUseCase3(unittest.TestCase):
    """Test of maximum mutual information (max MI, capacity) estimation 
    with optimization of weights of input distributions."""

    def test_equal_sizes(self):
        data = communicate(input_counts={'A': 5000, 'B': 5000},
                           output_values={'A': 0.0, 'B': 1.0})
        cap_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        cap_exact = 1.0
        self.assertAlmostEqual((cap_est - cap_exact)/cap_exact, 0, delta=RTOL)
        self.assertAlmostEqual(opt_ws['A'], 0.5, delta=ATOL)
        self.assertAlmostEqual(opt_ws['B'], 0.5, delta=ATOL)

    def test_unequal_sizes(self):
        data = communicate({'A': 2500, 'B': 5000},
                           {'A': 0.0, 'B': 1.0})
        cap_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        cap_exact = 1.0
        self.assertAlmostEqual((cap_est - cap_exact)/cap_exact, 0, delta=RTOL)
        self.assertAlmostEqual(opt_ws['A'], 0.5, delta=ATOL)
        self.assertAlmostEqual(opt_ws['B'], 0.5, delta=ATOL)

    def test_equal_sizes_partial_overlap(self):
        data = communicate({'A': 2500, 'B': 2500, 'C': 2500},
                           {'A': 0, 'B': 0, 'C': 1})
        cap_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        cap_exact = 1.0
        self.assertAlmostEqual((cap_est - cap_exact)/cap_exact, 0, delta=RTOL)
        self.assertAlmostEqual(opt_ws['C'], 0.5, delta=ATOL)

    def test_eight_gaussians(self):
        """This test case is patterned after a test included in the source code 
        featuring an article by Tudelska et al. (Scientific Reports, 2017), 
        DOI: doi.org/10.1038/s41598-017-16166-y ."""

        def _mu_sigma(i):
            return (1.5*(i + 1), (i + 1)/4. + 1)

        data = []
        for dist_i, count in enumerate(8*[5000]):
            data.extend([(dist_i, [values]) 
                         for values in norm(*_mu_sigma(dist_i)).rvs(count)])

        cap_est, opt_ws = wke(data).calculate_maximized_mi(k=NN_K)
        largest_2_opt_ws_indices = list(map(lambda kv: kv[0], 
                sorted(opt_ws.items(), key=lambda kv: kv[1], reverse=True)[0:2]))
        cap_accur, largest_2_opt_ws_indices_accur = 1.14425, [0,7]
        self.assertAlmostEqual((cap_est - cap_accur)/cap_accur, 0, delta=RTOL)
        self.assertEqual(largest_2_opt_ws_indices, largest_2_opt_ws_indices_accur)

    def test_8gaussians3d_snake(self):
        data = []
        for dist_i, count in enumerate(8*[5000]):
            mu, sigma = dist_i + 1, 33 + 3*dist_i
            means3 = [mu**2.5, 50*cos(mu/0.75), 200*sin(mu/1.5)]
            covar3 = sigma**2*identity(3);
            for v in multinorm(means3, covar3).rvs(count):
                data.append((dist_i, v))
        cap_est = wke(data).calculate_maximized_mi(k=NN_K)
        cap_accur = 1.91191 # calculated in Mathematica for continuous distributions
        self.assertAlmostEqual((cap_est[0] - cap_accur)/cap_accur, 0, delta=RTOL)

    def test_8gaussians3d_box(self):
        corners = list(product(range(2),repeat=3))
        data = []
        for i, count in enumerate(8*[5000]):
            means3, sigma = (corners[i], 0.25 + ((i + 1)/8 + sum(corners[i]))/10)
            covar3 = sigma**2*identity(3);
            for v in multinorm(means3, covar3).rvs(count):
                data.append((i, v))
        cap_est = wke(data).calculate_maximized_mi(k=NN_K)
        cap_accur = 1.8035 # calculated in Mathematica for continuous distributions
        self.assertAlmostEqual((cap_est[0] - cap_accur)/cap_accur, 0, delta=RTOL)

if __name__ == '__main__':
    unittest.main()
