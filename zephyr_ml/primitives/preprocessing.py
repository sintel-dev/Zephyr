"""
Preprocessing functions
"""

import sklearn.model_selection


def train_test_split(
    X,
    y,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """
    Wrapper over sklearn.model_selection.train_test_split()
    Used to split only 2 arrays at once: X (features) and y (labels)

    Split arrays or matrices into random train and test subsets.
    """
    return sklearn.model_selection.train_test_split(
        X, y, test_size, train_size, random_state, shuffle, stratify
    )
