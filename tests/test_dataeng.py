import unittest
from cce.dataeng import normalise

# Test data sets
# **************
big_distinct_small_spread = [(1, [1e9, 1e9]), (1, [1e9+1, 1e9+1]), (2, [1e9+2, 1e9+2]), (2, [1e9+3, 1e9+3])]
big_distinct_big_spread = [("1", [1e9]), ("1", [8e8]), (2, [-1e4]), (2, [-3e5]), ("3", [0])]
small_distinct = [("4", [0.001]), ("4", [0.002]), (2, [0.005]), (2, [0.006]), ("3", [-0.00001])]


distinct_data_sets = [big_distinct_small_spread, big_distinct_big_spread, small_distinct]


# Tests
# *****

class NormalisationTests(unittest.TestCase):
    def test_if_normalised(self):
        """Tests if all the data points are in the interval [0, 1]"""
        for data_set in distinct_data_sets:
            nor = normalise(data_set)
            for point in nor:
                for coord in point[1]:
                    self.assertTrue(0 <= coord <= 1)

    def test_if_ends_reached(self):
        """Tests if 0 and 1 are reached since we do not want to allow the normalisation to e.g. [0, 0.0001]"""
        for data_set in distinct_data_sets:
            nor = normalise(data_set)
            mn, mx = 1e9, -1e9

            for point in nor:
                mn = min(mn, *point[1])
                mx = max(mx, *point[1])

            self.assertAlmostEqual(mn, 0)
            self.assertAlmostEqual(mx, 1)


class StirringTests(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
