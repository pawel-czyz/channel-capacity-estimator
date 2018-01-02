import unittest
from cce.estimator import WeightedKraskovEstimator


class TestEstimatorSimpleMethods(unittest.TestCase):
    def test_k_type_1(self):
        """Tests if TypeError is raised. Type: string"""
        e = WeightedKraskovEstimator()
        with self.assertRaises(TypeError):
            e.k = "aaa"

    def test_k_type_2(self):
        """Tests if TypeError is raised. Type: float"""
        e = WeightedKraskovEstimator()
        with self.assertRaises(TypeError):
            e.k = 12.0

    def test_k_value_1(self):
        """Tests if ValueError is raised. Value: 0"""
        e = WeightedKraskovEstimator()
        with self.assertRaises(ValueError):
            e.k = 0

    def test_k_value_2(self):
        """Tests if ValueError is raised. Value: -5"""
        e = WeightedKraskovEstimator()
        with self.assertRaises(ValueError):
            e.k = -5

    def test_k_is_set(self):
        """Tests if k value is set properly."""
        for k in [1, 10, 13, 20, 100, 120]:
            e = WeightedKraskovEstimator(k=k)

            self.assertEqual(k, e.k)


if __name__ == '__main__':
    unittest.main()
