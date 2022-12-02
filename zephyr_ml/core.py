"""Zephyr Core module.

This module defines the Zephyr Class, which is responsible for the
model training and inference with the underlying MLBlocks pipelines.
"""
import json
import logging
import os
import pickle
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
from mlblocks import MLPipeline
from sklearn import metrics

LOGGER = logging.getLogger(__name__)


_REGRESSION_METRICS = {
    'mae': metrics.mean_absolute_error,
    'mse': metrics.mean_squared_error,
    'r2': metrics.r2_score,
}

_CLASSIFICATION_METRICS = {
    'accuracy': metrics.accuracy_score,
    'f1': metrics.f1_score,
    'recall': metrics.recall_score,
    'precision': metrics.precision_score,
}

METRICS = _CLASSIFICATION_METRICS


class Zephyr:
    """Zephyr Class.

    The Zephyr Class provides the main machine learning pipeline functionalities
    of Zephyr and is responsible for the interaction with the underlying
    MLBlocks pipelines.

    Args:
        pipeline (str, dict or MLPipeline):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """
    DEFAULT_PIPELINE = 'xgb'

    def _get_mlpipeline(self):
        pipeline = self._pipeline
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)

        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)

        return mlpipeline

    def __init__(self, pipeline: Union[str, dict, MLPipeline] = None,
                 hyperparameters: dict = None):
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self._pipeline == other._pipeline and
            self._hyperparameters == other._hyperparameters and
            self._fitted == other._fitted
        )

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], **kwargs):
        """Fit the pipeline to the given data.

        Args:
            X (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                the feature matrix.
            y (Series or ndarray):
                Target data, passed as a ``pandas.Series`` or ``numpy.ndarray``
                containing the target values.
        """
        if not self._fitted:
            self._mlpipeline = self._get_mlpipeline()

        self._mlpipeline.fit(X, y, **kwargs)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the pipeline to the given data.

        Args:
            X (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                the feature matrix.

        Returns:
            Series or ndarray:
                Predictions to the input data.
        """
        return self._mlpipeline.predict(X)

    def fit_predict(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                    **kwargs) -> pd.Series:
        """Fit the pipeline to the data and then predict targets.

        This method is functionally equivalent to calling ``fit(X, y)``
        and later on ``predict(X)`` but with the difference that
        here the ``MLPipeline`` is called only once, using its ``fit``
        method, and the output is directly captured without having
        to execute the whole pipeline again during the ``predict`` phase.

        Args:
            X (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                the feature matrix.
            y (Series or ndarray):
                Target data, passed as a ``pandas.Series`` or ``numpy.ndarray``
                containing the target values.

        Returns:
            Series or ndarray:
                Predictions to the input data.
        """
        if not self._fitted:
            self._mlpipeline = self._get_mlpipeline()

        result = self._mlpipeline.fit(X, y, output_='default', **kwargs)
        self._fitted = True

        return result

    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], fit: bool = False,
                 train_X: pd.DataFrame = None, train_y: Union[pd.Series, np.ndarray] = None,
                 metrics: List[str] = METRICS) -> pd.Series:
        """Evaluate the performance of the pipeline.

        Args:
            X (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                the feature matrix.
            y (Series or ndarray):
                Target data, passed as a ``pandas.Series`` or ``numpy.ndarray``
                containing the target values.
            fit (bool):
                Whether to fit the pipeline before evaluating it.
                Defaults to ``False``.
            train_X (DataFrame):
                Training data, passed as a ``pandas.DataFrame`` containing
                the feature matrix.
                If not given, the pipeline is fitted on ``X``.
            train_y (Series or ndarray):
                Target data used for training, passed as a ``pandas.Series`` or
                ``numpy.ndarray`` containing the target values.
            metrics (list):
                List of metrics to used passed as a list of strings.
                If not given, it defaults to all the metrics.

        Returns:
            Series:
                ``pandas.Series`` containing one element for each
                metric applied, with the metric name as index.
        """
        if not fit:
            method = self._mlpipeline.predict
        else:
            if not self._fitted:
                mlpipeline = self._get_mlpipeline()
            else:
                mlpipeline = self._mlpipeline

            if train_X is not None and train_y is not None:
                # fit first and then predict
                mlpipeline.fit(train_X, train_y)
                method = mlpipeline.predict
            else:
                # fit and predict at once
                method = partial(mlpipeline.fit, y=y, output_='default')

        result = method(X)

        scores = {
            metric: METRICS[metric](y, result)
            for metric in metrics
        }

        return pd.Series(scores)

    def save(self, path: str):
        """Save this object using pickle.

        Args:
            path (str):
                Path to the file where the serialization of
                this object will be stored.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path: str):
        """Load an Zephyr instance from a pickle file.

        Args:
            path (str):
                Path to the file where the instance has been
                previously serialized.

        Returns:
            Orion

        Raises:
            ValueError:
                If the serialized object is not a Zephyr instance.
        """
        with open(path, 'rb') as pickle_file:
            zephyr = pickle.load(pickle_file)
            if not isinstance(zephyr, cls):
                raise ValueError('Serialized object is not a Zephyr instance')

            return zephyr
