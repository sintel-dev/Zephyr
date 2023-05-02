"""
Postprocessing functions.
"""
import logging

import numpy as np
import sklearn

LOGGER = logging.getLogger(__name__)

METRICS = {
    "accuracy": sklearn.metrics.accuracy_score,
    "precision": sklearn.metrics.precision_score,
    "recall": sklearn.metrics.recall_score,
    "f1": sklearn.metrics.f1_score,
}


class FindThreshold:
    """Find Optimal Threshold.

    This class find the optimal threshold value that produces
    the highest metric score. In the fit phase, it detects
    the best threshold based on the given metric. In the
    produce phase, it applies the found threshold on the
    predicted values.

    This is intended for classification problems.

    Args:
        metric (str):
            String representing which metric to use.
    """

    def __init__(self, metric='f1'):
        self._metric = 'f1'
        self._threshold = None

    def fit(self, y_true, y_pred):
        """Find the threshold that obtains the best metric value.

        Args:
            y_true (Series or ndarray):
                ``pandas.Series`` or ``numpy.ndarray`` ground truth target values.
            y_pred (Series or ndarray):
                ``pandas.Series`` or ``numpy.ndarray`` predicted target valeus.
        """
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 1]

        RANGE = np.arange(0, 1, 0.01)

        scores = list()
        scorer = METRICS[self._metric]
        for thresh in RANGE:
            y = [1 if x else 0 for x in y_pred > thresh]
            scores.append(scorer(y_true, y))

        threshold = RANGE[np.argmax(scores)]
        LOGGER.info(f'best threshold found at {threshold}')

        self._threshold = threshold
        self._scores = scores

    def apply_threshold(self, y_pred):
        """Apply threshold on predicted values.

        Args:
            y_pred (Series):
                ``pandas.Series`` predicted target valeus.

        Return:
            tuple:
                * list of predicted target valeus in binary codes.
                * detected float value for threshold.
                * list of scores obtained at each threshold.
        """
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 1]

        binary = [1 if x else 0 for x in y_pred > self._threshold]
        return binary, self._threshold, self._scores
