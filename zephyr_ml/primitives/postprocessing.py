"""
Postprocessing functions.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn import metrics

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

    def __init__(self, metric="f1"):
        self._metric = "f1"
        self._threshold = None

    def fit(self, y_true, y_proba):
        """Find the threshold that obtains the best metric value.

        Args:
            y_true (Series or ndarray):
                ``pandas.Series`` or ``numpy.ndarray`` ground truth target values.
            y_proba (Series or ndarray):
                ``pandas.Series`` or ``numpy.ndarray`` predicted target values' probabilities.
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]

        RANGE = np.arange(0, 1, 0.01)

        scores = list()
        scorer = METRICS[self._metric]
        for thresh in RANGE:
            y = [1 if x else 0 for x in y_proba > thresh]
            scores.append(scorer(y_true, y))

        threshold = RANGE[np.argmax(scores)]
        LOGGER.info(f"best threshold found at {threshold}")

        self._threshold = threshold
        self._scores = scores

    def apply_threshold(self, y_proba):
        """Apply threshold on predicted values.

        Args:
            y_pred (Series):
                ``pandas.Series`` predicted target values' probabilities.

        Return:
            tuple:
                * list of predicted target valeus in binary codes.
                * detected float value for threshold.
                * list of scores obtained at each threshold.
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]

        binary = [1 if x else 0 for x in y_proba > self._threshold]
        return binary, self._threshold, self._scores


def confusion_matrix(
        y_true,
        y_pred,
        labels=None,
        sample_weight=None,
        normalize=None):
    conf_matrix = metrics.confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=normalize
    )
    fig = plt.figure()
    ax = fig.add_axes(sns.heatmap(conf_matrix, annot=True, cmap="Blues"))

    ax.set_title("Confusion Matrix\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(["False", "True"])
    ax.yaxis.set_ticklabels(["False", "True"])

    return conf_matrix, fig


def roc_auc_score_and_curve(
    y_true, y_proba, pos_label=None, sample_weight=None, drop_intermediate=True
):
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    fpr, tpr, _ = metrics.roc_curve(
        y_true,
        y_proba,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )
    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(
        y_true,
        ns_probs,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )

    auc = metrics.roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(1, 1)

    ax.plot(fpr, tpr, "ro")
    ax.plot(fpr, tpr)
    ax.plot(ns_fpr, ns_tpr, linestyle="--", color="green")

    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("AUC: %.3f" % auc)

    return auc, fig
