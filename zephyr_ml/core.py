from zephyr_ml.metadata import get_default_es_type_kwargs
from zephyr_ml.entityset import _create_entityset, VALIDATE_DATA_FUNCTIONS
from zephyr_ml.labeling import get_labeling_functions, LABELING_FUNCTIONS

import composeml as cp
from inspect import getfullargspec
import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from mlblocks import MLPipeline


class Zephyr:

    def __init__(self):
        self.entityset = None
        self.labeling_function = None
        self.label_times = None
        self.pipeline = None
        self.pipeline_hyperparameters = None
        self.feature_matrix_and_labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_entityset_types(self):
        """
        Returns the supported entityset types (PI/SCADA) and the required dataframes and their columns
        """
        return VALIDATE_DATA_FUNCTIONS.keys()

    def create_entityset(self, data_paths, es_type, new_kwargs_mapping=None):
        """
        Generate an entityset

        Args:
        data_paths ( dict ): Dictionary mapping entity names to the pandas
        dataframe for that that entity
        es_type (str): type of signal data , either SCADA or PI
        new_kwargs_mapping ( dict ): Updated keyword arguments to be used
        during entityset creation
        Returns:
        featuretools.EntitySet that contains the data passed in and
        their relationships
        """
        entityset = _create_entityset(data_paths, es_type, new_kwargs_mapping)
        self.entityset = entityset
        return self.entityset

    def get_entityset(self):
        if self.entityset is None:
            raise ValueError("No entityset has been created or set in this instance.")

        return self.entityset

    def set_entityset(self, entityset, es_type, new_kwargs_mapping=None):
        dfs = entityset.to_dictionary()

        validate_func = VALIDATE_DATA_FUNCTIONS[es_type]
        validate_func(dfs, new_kwargs_mapping)

        self.entityset = entityset

    def get_predefined_labeling_functions(self):
        return get_labeling_functions()

    def set_labeling_functions(self, name=None, func=None):
        if name is not None:
            if name in LABELING_FUNCTIONS:
                self.labeling_function = LABELING_FUNCTIONS[name]
            else:
                raise ValueError(
                    f"Unrecognized name argument:{name}. Call get_predefined_labeling_functions to view predefined labeling functions"
                )
        elif func is not None:
            if callable(func):
                self.labeling_function = func
            else:
                raise ValueError(f"Custom function is not callable")
        raise ValueError("No labeling function given.")

    def generate_labeling_times(
        self, num_samples=-1, subset=None, column_map={}, verbose=False, **kwargs
    ):
        assert self.entityset is not None
        assert self.labeling_function is not None

        labeling_function, df, meta = self.labeling_function(self.entityset, column_map)

        data = df
        if isinstance(subset, float) or isinstance(subset, int):
            data = data.sample(subset)

        target_entity_index = meta.get("target_entity_index")
        time_index = meta.get("time_index")
        thresh = kwargs.get("thresh") or meta.get("thresh")
        window_size = kwargs.get("window_size") or meta.get("window_size")
        label_maker = cp.LabelMaker(
            labeling_function=labeling_function,
            target_dataframe_name=target_entity_index,
            time_index=time_index,
            window_size=window_size,
        )

        kwargs = {**meta, **kwargs}
        kwargs = {
            k: kwargs.get(k)
            for k in set(getfullargspec(label_maker.search)[0])
            if kwargs.get(k) is not None
        }
        label_times = label_maker.search(
            data.sort_values(time_index), num_samples, verbose=verbose, **kwargs
        )
        if thresh is not None:
            label_times = label_times.threshold(thresh)

        self.label_times = label_times

        return label_times, meta

    def plot_label_times(self):
        assert self.label_times is not None
        cp.label_times.plots.LabelPlots(self.label_times).distribution()

    def generate_features(self, **kwargs):

        feature_matrix, features = ft.dfs(
            entityset=self.entityset, cutoff_time=self.label_times, **kwargs
        )
        self.feature_matrix_and_labels = self._clean_feature_matrix(feature_matrix)
        self.features = features
        return feature_matrix, features

    def get_feature_matrix_and_labels(self):
        return self.feature_matrix_and_labels

    def set_feature_matrix_and_labels(self, feature_matrix, label_col_name="label"):
        assert label_col_name in feature_matrix.columns
        self.feature_matrix_and_labels = self._clean_feature_matrix(
            feature_matrix, label_col_name=label_col_name
        )

    def generate_train_test_split(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        feature_matrix, labels = self.feature_matrix_and_labels
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            labels,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return

    def set_train_test_split(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_train_test_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_predefined_pipelines(self):
        pass

    def set_pipeline(self, pipeline, pipeline_hyperparameters):
        self.pipeline = self._get_mlpipeline(pipeline, pipeline_hyperparameters)
        self.pipeline_hyperparameters = pipeline_hyperparameters

    def get_pipeline(self):
        return self.pipeline

    def fit(
        self, X=None, y=None, visual=False, **kwargs
    ):  # kwargs indicate the parameters of the current pipeline
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        if visual:
            outputs_spec, visual_names = self._get_outputs_spec(False)
        else:
            outputs_spec = None

        outputs = self.pipeline.fit(X, y, output_=outputs_spec, **kwargs)

        if visual and outputs is not None:
            return dict(zip(visual_names, outputs))

    def predict(self, X=None, visual=False, **kwargs):
        if X is None:
            X = self.X_test
        if visual:
            outputs_spec, visual_names = self._get_outputs_spec()
        else:
            outputs_spec = "default"

        outputs = self.pipeline.predict(X, output_=outputs_spec, **kwargs)

        if visual and visual_names:
            prediction = outputs[0]
            return prediction, dict(zip(visual_names, outputs[-len(visual_names) :]))

        return outputs

    def evaluate(self, X=None, y=None, metrics=None):
        result = self.pipeline.predict(X)

        pass

    def _validate_step(self, **kwargs):
        for key, value in kwargs:
            assert (value is not None, f"{key} has not been set or created")

    def _clean_feature_matrix(self, feature_matrix, label_col_name="label"):
        labels = feature_matrix.pop(label_col_name)

        count_cols = feature_matrix.filter(like="COUNT").columns
        feature_matrix[count_cols] = feature_matrix[count_cols].apply(
            lambda x: x.astype(np.int64)
        )

        string_cols = feature_matrix.select_dtypes(include="category").columns
        feature_matrix = pd.get_dummies(feature_matrix, columns=string_cols)

        return feature_matrix, labels

    def _get_mlpipeline(self, pipeline, hyperparameters):
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)

        mlpipeline = MLPipeline(pipeline)
        if hyperparameters:
            mlpipeline.set_hyperparameters(hyperparameters)

        return mlpipeline

    def _get_outputs_spec(self, default=True):
        outputs_spec = ["default"] if default else []

        try:
            visual_names = self.pipeline.get_output_names("visual")
            outputs_spec.append("visual")
        except ValueError:
            visual_names = []

        return outputs_spec, visual_names
