import unittest

import math

from k_means import kmeans


class KmeansTestCase(unittest.TestCase):
    """Tests for `kmeans.py`."""

    def test_euclid(self):
        c = kmeans.Centroid()
        c.location = [6, 7, 2]
        p1 = [1, 2, 5]

        test_dist = math.sqrt(59)

        self.assertEqual(kmeans.euclid_dist(p1, c), test_dist)


if __name__ == '__main__':
    unittest.main()
