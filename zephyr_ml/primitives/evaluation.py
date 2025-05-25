"""
Evaluation metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


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
