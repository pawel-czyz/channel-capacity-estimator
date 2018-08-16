import unittest
from itertools import product
from numpy import cos, sin, log2
from scipy.stats import multivariate_normal as multinorm
from cce.estimator import WeightedKraskovEstimator as wke
from tests.plain_kraskov import calculate_mi as plain_calculate_mi
from tests.noisy_channel import communicate

NN_K = 15
ATOL = 0.02
RTOL = 0.04

class TestUseCase1(unittest.TestCase):
    """Tests of mutual information (MI) estimation for input distributions
    without imposed weights."""

    def test_equal_sizes_trivial_separate(self):
        data = communicate(input_counts={'0': 10000, '1': 10000}, 
                           output_values={'0': 0.0, '1': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = 1.0
        self.assertAlmostEqual((mi_est1 - mi_exact)/mi_exact, 0, delta=RTOL)
        self.assertAlmostEqual((mi_est1 - mi_est2)/mi_est2, 0, delta=RTOL)

    def test_equal_sizes_trivial_overlap(self):
        data = communicate({'0': 10000, '1': 10000},
                           {'0': 0.0, '1': 0.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = 0.0
        self.assertAlmostEqual(mi_est1, mi_exact, delta=ATOL)
        self.assertAlmostEqual(mi_est1, mi_est2, delta=ATOL)

    def test_unequal_sizes(self):
        data = communicate({'0': 5000, '1': 10000},
                           {'0': 0.0, '1': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = -(log2(2/3)*2/3 + log2(1/3)*1/3)
        self.assertAlmostEqual((mi_est1 - mi_exact)/mi_exact, 0, delta=RTOL)
        self.assertAlmostEqual((mi_est1 - mi_est2)/mi_exact, 0, delta=RTOL)

    def test_equal_sizes_partial_overlap(self):
        data = communicate({'0': 5000, '1': 5000, '2': 5000},
                           {'0': 0.0, '1': 0.0, '2': 1.0})
        mi_est1 = wke(data).calculate_mi(k=NN_K)
        mi_est2 = plain_calculate_mi(data=data, k=NN_K)
        mi_exact = -(log2(2/3)*2/3 + log2(1/3)*1/3)
        self.assertAlmostEqual((mi_est1 - mi_exact)/mi_exact, 0, delta=RTOL)
        self.assertAlmostEqual((mi_est1 - mi_est2)/mi_est2, 0, delta=RTOL)

    def test_8gaussians3d_snake(self):
        data = []
        for dist_i, count in enumerate(8*[5000]):
            mu, sigma = dist_i+1, 33 + 3*dist_i
            means3 = [mu**2.5, 50*cos(mu/0.75), 200*sin(mu/1.5)]
            covar3 = [[sigma**2, 0, 0],
                      [0, sigma**2, 0],
                      [0, 0, sigma**2]]
            for v in multinorm(means3, covar3).rvs(count):
                data.append((dist_i, v))
        mi_est = wke(data).calculate_mi(k=NN_K)
        mi_accur = 1.85959 # calculated in Mathematica for continuous distributions
        self.assertAlmostEqual((mi_est - mi_accur)/mi_accur, 0, delta=RTOL)

    def test_8gaussians3d_box(self):
        box = list(product(range(2),repeat=3))
        data = []
        for i, count in enumerate(8*[5000]):
            means3, sigma = (box[i], 0.25 + ((i + 1)/8 + sum(box[i]))/10)
            covar3 = [[sigma**2, 0, 0],
                      [0, sigma**2, 0],
                      [0, 0, sigma**2]]
            for v in multinorm(means3, covar3).rvs(count):
                data.append((i, v))
        mi_est = wke(data).calculate_mi(k=NN_K)
        mi_accur = 1.72251 # calculated in Mathematica for continuous distributions
        self.assertAlmostEqual((mi_est - mi_accur)/mi_accur, 0, delta=RTOL)


if __name__ == '__main__':
    unittest.main()
