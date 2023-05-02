from unittest import TestCase

import numpy as np

from zephyr_ml.primitives.postprocessing import FindThreshold


class FindThresholdTest(TestCase):

    y = np.array([1, 1, 0, 0, 1, 1, 1, 0])
    y_hat_1d = np.array([0.8, 0.9, 0.6, 0.5, 0.85, 0.7, 0.95, 0.2])
    y_hat_2d = np.array([[0.2, 0.8],
                         [0.1, 0.9],
                         [0.4, 0.6],
                         [0.5, 0.5],
                         [0.15, 0.85],
                         [0.3, 0.7],
                         [0.05, 0.95],
                         [0.8, 0.2]])

    def _run(self, y, y_hat, value):
        threshold = FindThreshold()
        threshold.fit(y, y_hat)

        assert threshold._threshold == value
        binary_y_hat, detected_threshold, scores = threshold.apply_threshold(y_hat)
        np.testing.assert_allclose(binary_y_hat, y)

    def test_1d(self):
        self._run(self.y, self.y_hat_1d, 0.6)

    def test_2d(self):
        self._run(self.y, self.y_hat_2d, 0.6)
