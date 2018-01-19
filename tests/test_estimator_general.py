import unittest
from cce.estimator import WeightedKraskovEstimator

data_1d = [("bunny", [i]) for i in range(7)] + [("teddy", [i]) for i in range(10)]


class TestEstimator(unittest.TestCase):
    def test_load(self):
        """Dokończyć test czy dane są poprawnie ładowane"""
        wke = WeightedKraskovEstimator(data=data_1d)

        self.assertTrue(wke._data_loaded)
        self.assertTrue(wke._new_data_loaded)
        self.assertTrue(wke._number_of_labels, 2)
        self.assertTrue(wke._number_of_points_total, len(data_1d))
        wke.calculate_neighborhood(10)

        # TODO
        # self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
