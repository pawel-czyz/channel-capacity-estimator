import unittest
from cce.preprocess import normalize

LARGE_VALUES_SMALL_SPREAD = [('1', [1e9, 1e9]),
                             ('1', [1e9+1, 1e9+1]),
                             ('2', [1e9+2, 1e9+2]),
                             ('2', [1e9+3, 1e9+3])]
LARGE_VALUES_LARGE_SPREAD = [('1', [1e9]),
                             ('1', [8e8]),
                             ('2', [-1e4]),
                             ('2', [-3e5]),
                             ('3', [0])]
SMALL_VALUES = [('4', [0.001]),
                ('4', [0.002]),
                ('2', [0.005]),
                ('2', [0.006]),
                ('3', [-0.00001])]

DATA_SETS = [LARGE_VALUES_SMALL_SPREAD, LARGE_VALUES_LARGE_SPREAD, SMALL_VALUES]


class NormalizationTests(unittest.TestCase):

    def test_if_normalized(self):
        """Test if all the data points are in the interval [0, 1]."""
        for data_set in DATA_SETS:
            nor = normalize(data_set)
            for point in nor:
                for coord in point[1]:
                    self.assertTrue(0 <= coord <= 1)

    def test_if_range_is_0_1(self):
        """Test if lower bound of 0 and upper bound of 1 are reached."""
        for data_set in DATA_SETS:
            nor = normalize(data_set)
            min_, max_ = 1e9, -1e9

            for point in nor:
                min_ = min(min_, *point[1])
                max_ = max(max_, *point[1])

            self.assertAlmostEqual(min_, 0)
            self.assertAlmostEqual(max_, 1)


if __name__ == '__main__':
    unittest.main()
