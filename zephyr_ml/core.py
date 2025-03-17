# from zephyr_ml.entityset import _create_entityset, VALIDATE_DATA_FUNCTIONS
# from zephyr_ml.labeling import get_labeling_functions, LABELING_FUNCTIONS
# from zephyr_ml.entityset import _create_entityset, VALIDATE_DATA_FUNCTIONS
from zephyr_ml.labeling import (
    get_labeling_functions,
    LABELING_FUNCTIONS,
)
import composeml as cp
from inspect import getfullargspec
import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from mlblocks import MLPipeline, MLBlock, get_primitives_paths, add_primitives_path
from itertools import chain

DEFAULT_METRICS = [
    "sklearn.metrics.accuracy_score",
    "sklearn.metrics.precision_score",
    "sklearn.metrics.f1_score",
    "sklearn.metrics.recall_score",
]


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

    def set_labeling_function(self, name=None, func=None):
        print(f"labeling fucntion name {name}")
        if name is not None:
            labeling_fn_map = get_labeling_functions_map()
            if name in labeling_fn_map:
                self.labeling_function = labeling_fn_map[name]
                return
            else:
                raise ValueError(
                    f"Unrecognized name argument:{name}. Call get_predefined_labeling_functions to view predefined labeling functions"
                )
        elif func is not None:
            if callable(func):
                self.labeling_function = func
                return
            else:
                raise ValueError(f"Custom function is not callable")
        raise ValueError("No labeling function given.")

    def generate_label_times(
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
        print(feature_matrix)
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

    def set_pipeline(self, pipeline, pipeline_hyperparameters=None):
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
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        context_0 = self.pipeline.predict(X, output_=0)
        y_proba = context_0["y_pred"]
        y_pred = self.pipeline.predict(start_=1, **context_0)
        if metrics is None:
            metrics = DEFAULT_METRICS

        for metric in metrics:
            metric_primitive = self._get_ml_primitive(metric)
            print(metric_primitive)
            res = metric_primitive.produce(y_pred=y_pred, y_proba=y_proba, y_true=y)
            print(metric_primitive.name, res)

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

    def _get_mlpipeline(self, pipeline, hyperparameters=None):
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)

        mlpipeline = MLPipeline(pipeline)
        if hyperparameters:
            mlpipeline.set_hyperparameters(hyperparameters)

        return mlpipeline

    def _get_ml_primitive(self, primitive, hyperparameters=None):
        if isinstance(primitive, str) and os.path.isfile(primitive):
            with open(primitive) as json_file:
                primitive = json.load(json_file)
        mlprimitive = MLBlock(primitive)

        if hyperparameters:
            mlprimitive.set_hyperparameters(hyperparameters)
        return mlprimitive

    def _get_outputs_spec(self, default=True):
        outputs_spec = ["default"] if default else []

        try:
            visual_names = self.pipeline.get_output_names("visual")
            outputs_spec.append("visual")
        except ValueError:
            visual_names = []

        return outputs_spec, visual_names


def get_labeling_functions_map():
    functions = {}
    for function in LABELING_FUNCTIONS:
        name = function.__name__
        functions[name] = function
    return functions


import copy


# Default EntitySet keyword arguments for entities
DEFAULT_ES_KWARGS = {
    "alarms": {
        "index": "_index",
        "make_index": True,
        "time_index": "DAT_START",
        "secondary_time_index": {"DAT_END": ["IND_DURATION"]},
        "logical_types": {
            "COD_ELEMENT": "categorical",  # turbine id
            "DAT_START": "datetime",  # start
            "DAT_END": "datetime",  # end
            "IND_DURATION": "double",  # duration
            "COD_ALARM": "categorical",  # alarm code
            "COD_ALARM_INT": "categorical",  # international alarm code
            "DES_NAME": "categorical",  # alarm name
            "DES_TITLE": "categorical",  # alarm description
            "COD_STATUS": "categorical",  # status code
        },
    },
    "stoppages": {
        "index": "_index",
        "make_index": True,
        "time_index": "DAT_START",
        "secondary_time_index": {"DAT_END": ["IND_DURATION", "IND_LOST_GEN"]},
        "logical_types": {
            "COD_ELEMENT": "categorical",  # turbine id
            "DAT_START": "datetime",  # start
            "DAT_END": "datetime",  # end
            "DES_WO_NAME": "natural_language",  # work order name
            "DES_COMMENTS": "natural_language",  # work order comments
            "COD_WO": "integer_nullable",  # stoppage code
            "IND_DURATION": "double",  # duration
            "IND_LOST_GEN": "double",  # generation loss
            "COD_ALARM": "categorical",  # alarm code
            "COD_CAUSE": "categorical",  # stoppage cause
            "COD_INCIDENCE": "categorical",  # incidence code
            "COD_ORIGIN": "categorical",  # origin code
            "DESC_CLASS": "categorical",  # ????
            "COD_STATUS": "categorical",  # status code
            "COD_CODE": "categorical",  # stoppage code
            "DES_DESCRIPTION": "natural_language",  # stoppage description
            "DES_TECH_NAME": "categorical",  # turbine technology
        },
    },
    "notifications": {
        "index": "_index",
        "make_index": True,
        "time_index": "DAT_POSTING",
        "secondary_time_index": {"DAT_MALF_END": ["IND_BREAKDOWN_DUR"]},
        "logical_types": {
            "COD_ELEMENT": "categorical",  # turbine id
            "COD_ORDER": "categorical",
            "IND_QUANTITY": "double",
            "COD_MATERIAL_SAP": "categorical",
            "DAT_POSTING": "datetime",
            "COD_MAT_DOC": "categorical",
            "DES_MEDIUM": "categorical",
            "COD_NOTIF": "categorical",
            "DAT_MALF_START": "datetime",
            "DAT_MALF_END": "datetime",
            "IND_BREAKDOWN_DUR": "double",
            "FUNCT_LOC_DES": "categorical",
            "COD_ALARM": "categorical",
            "DES_ALARM": "categorical",
        },
    },
    "work_orders": {
        "index": "COD_ORDER",
        "time_index": "DAT_BASIC_START",
        "secondary_time_index": {"DAT_VALID_END": []},
        "logical_types": {
            "COD_ELEMENT": "categorical",
            "COD_ORDER": "categorical",
            "DAT_BASIC_START": "datetime",
            "DAT_BASIC_END": "datetime",
            "COD_EQUIPMENT": "categorical",
            "COD_MAINT_PLANT": "categorical",
            "COD_MAINT_ACT_TYPE": "categorical",
            "COD_CREATED_BY": "categorical",
            "COD_ORDER_TYPE": "categorical",
            "DAT_REFERENCE": "datetime",
            "DAT_CREATED_ON": "datetime",
            "DAT_VALID_END": "datetime",
            "DAT_VALID_START": "datetime",
            "COD_SYSTEM_STAT": "categorical",
            "DES_LONG": "natural_language",
            "COD_FUNCT_LOC": "categorical",
            "COD_NOTIF_OBJ": "categorical",
            "COD_MAINT_ITEM": "categorical",
            "DES_MEDIUM": "natural_language",
            "DES_FUNCT_LOC": "categorical",
        },
    },
    "turbines": {
        "index": "COD_ELEMENT",
        "logical_types": {
            "COD_ELEMENT": "categorical",
            "TURBINE_PI_ID": "categorical",
            "TURBINE_LOCAL_ID": "categorical",
            "TURBINE_SAP_COD": "categorical",
            "DES_CORE_ELEMENT": "categorical",
            "SITE": "categorical",
            "DES_CORE_PLANT": "categorical",
            "COD_PLANT_SAP": "categorical",
            "PI_COLLECTOR_SITE_NAME": "categorical",
            "PI_LOCAL_SITE_NAME": "categorical",
        },
    },
}

DEFAULT_ES_TYPE_KWARGS = {
    "pidata": {
        "index": "_index",
        "make_index": True,
        "time_index": "time",
        "logical_types": {"time": "datetime", "COD_ELEMENT": "categorical"},
    },
    "scada": {
        "index": "_index",
        "make_index": True,
        "time_index": "TIMESTAMP",
        "logical_types": {"TIMESTAMP": "datetime", "COD_ELEMENT": "categorical"},
    },
    "vibrations": {
        "index": "_index",
        "make_index": True,
        "time_index": "timestamp",
        "logical_types": {
            "COD_ELEMENT": "categorical",
            "turbine_id": "categorical",
            "signal_id": "categorical",
            "timestamp": "datetime",
            "sensorName": "categorical",
            "sensorType": "categorical",
            "sensorSerial": "integer_nullable",
            "siteName": "categorical",
            "turbineName": "categorical",
            "turbineSerial": "integer_nullable",
            "configurationName": "natural_language",
            "softwareVersion": "categorical",
            "rpm": "double",
            "rpmStatus": "natural_language",
            "duration": "natural_language",
            "condition": "categorical",
            "maskTime": "datetime",
            "Mask Status": "natural_language",
            "System Serial": "categorical",
            "WPS-ActivePower-Average": "double",
            "WPS-ActivePower-Minimum": "double",
            "WPS-ActivePower-Maximum": "double",
            "WPS-ActivePower-Deviation": "double",
            "WPS-ActivePower-StartTime": "datetime",
            "WPS-ActivePower-StopTime": "datetime",
            "WPS-ActivePower-Counts": "natural_language",
            "Measured RPM": "double",
            "WPS-ActivePower": "double",
            "WPS-Gearoiltemperature": "double",
            "WPS-GeneratorRPM": "double",
            "WPS-PitchReference": "double",
            "WPS-RotorRPM": "double",
            "WPS-Windspeed": "double",
            "WPS-YawAngle": "double",
            "overload warning": "categorical",
            "bias warning": "categorical",
            "bias voltage": "double",
            "xValueOffset": "double",
            "xValueDelta": "double",
            "xValueUnit": "categorical",
            "yValueUnit": "categorical",
            "TotalCount-RPM0": "double",
            "TotalCount-RPM1": "double",
            "TotalCount-RPM2": "double",
            "TotalCount-RPM3": "double",
        },
    },
}


def get_mapped_kwargs(es_type, new_kwargs=None):
    if es_type not in DEFAULT_ES_TYPE_KWARGS.keys():
        raise ValueError("Unrecognized es_type argument: {}".format(es_type))
    mapped_kwargs = DEFAULT_ES_KWARGS.copy()
    mapped_kwargs.update({es_type: DEFAULT_ES_TYPE_KWARGS[es_type]})

    if new_kwargs is not None:
        if not isinstance(new_kwargs, dict):
            raise ValueError(
                "new_kwargs must be dictionary mapping entity name to dictionary "
                "with updated keyword arguments for EntitySet creation."
            )
        for entity in new_kwargs:
            if entity not in mapped_kwargs:
                raise ValueError(
                    'Unrecognized entity "{}" found in new keyword argument '
                    "mapping.".format(entity)
                )

            mapped_kwargs[entity].update(new_kwargs[entity])

    return mapped_kwargs


def get_default_es_type_kwargs():
    return copy.deepcopy(DEFAULT_ES_TYPE_KWARGS)


def get_es_types():
    return DEFAULT_ES_TYPE_KWARGS.keys()


def create_pidata_entityset(dfs, new_kwargs_mapping=None):
    """Generate an entityset for PI data datasets

    Args:
        data_paths (dict): Dictionary mapping entity names ('alarms', 'notifications',
            'stoppages', 'work_orders', 'pidata', 'turbines') to the pandas dataframe for
            that entity.
        **kwargs: Updated keyword arguments to be used during entityset creation
    """
    entity_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
    _validate_data(dfs, "pidata", entity_kwargs)

    es = _create_entityset(dfs, "pidata", entity_kwargs)
    es.id = "PI data"

    return es


def create_scada_entityset(dfs, new_kwargs_mapping=None):
    """Generate an entityset for SCADA data datasets

    Args:
        data_paths (dict): Dictionary mapping entity names ('alarms', 'notifications',
            'stoppages', 'work_orders', 'scada', 'turbines') to the pandas dataframe for
            that entity.
    """
    entity_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
    _validate_data(dfs, "scada", entity_kwargs)

    es = _create_entityset(dfs, "scada", entity_kwargs)
    es.id = "SCADA data"

    return es


def create_vibrations_entityset(dfs, new_kwargs_mapping=None):
    """Generate an entityset for Vibrations data datasets

    Args:
        data_paths (dict): Dictionary mapping entity names ('alarms', 'notifications',
            'stoppages', 'work_orders', 'vibrations', 'turbines') to the pandas
            dataframe for that entity. Optionally 'pidata' and 'scada' can be included.
    """
    entities = ["vibrations"]

    pidata_kwargs, scada_kwargs = {}, {}
    if "pidata" in dfs:
        pidata_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
        entities.append("pidata")
    if "scada" in dfs:
        scada_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
        entities.append("scada")

    entity_kwargs = {
        **pidata_kwargs,
        **scada_kwargs,
        **get_mapped_kwargs("vibrations", new_kwargs_mapping),
    }
    _validate_data(dfs, entities, entity_kwargs)

    es = _create_entityset(dfs, "vibrations", entity_kwargs)
    es.id = "Vibrations data"

    return es


def _validate_data(dfs, es_type, es_kwargs):
    """Validate data by checking for required columns in each entity"""
    if not isinstance(es_type, list):
        es_type = [es_type]

    entities = set(
        chain(
            [
                "alarms",
                "stoppages",
                "work_orders",
                "notifications",
                "turbines",
                *es_type,
            ]
        )
    )

    if set(dfs.keys()) != entities:
        missing = entities.difference(set(dfs.keys()))
        extra = set(dfs.keys()).difference(entities)
        msg = []
        if missing:
            msg.append("Missing dataframes for entities {}.".format(", ".join(missing)))
        if extra:
            msg.append(
                "Unrecognized entities {} included in dfs.".format(", ".join(extra))
            )

        raise ValueError(" ".join(msg))

    turbines_index = es_kwargs["turbines"]["index"]
    work_orders_index = es_kwargs["work_orders"]["index"]

    if work_orders_index not in dfs["work_orders"].columns:
        raise ValueError(
            'Expected index column "{}" missing from work_orders entity'.format(
                work_orders_index
            )
        )

    if work_orders_index not in dfs["notifications"].columns:
        raise ValueError(
            'Expected column "{}" missing from notifications entity'.format(
                work_orders_index
            )
        )

    if not dfs["work_orders"][work_orders_index].is_unique:
        raise ValueError(
            'Expected index column "{}" of work_orders entity is not '
            "unique".format(work_orders_index)
        )

    if turbines_index not in dfs["turbines"].columns:
        raise ValueError(
            'Expected index column "{}" missing from turbines entity'.format(
                turbines_index
            )
        )

    if not dfs["turbines"][turbines_index].is_unique:
        raise ValueError(
            'Expected index column "{}" of turbines entity is not unique.'.format(
                turbines_index
            )
        )

    for entity, df in dfs.items():
        if turbines_index not in df.columns:
            raise ValueError(
                'Turbines index column "{}" missing from data for {} entity'.format(
                    turbines_index, entity
                )
            )

        time_index = es_kwargs[entity].get("time_index", False)
        if time_index and time_index not in df.columns:
            raise ValueError(
                'Missing time index column "{}" from {} entity'.format(
                    time_index, entity
                )
            )

        secondary_time_indices = es_kwargs[entity].get("secondary_time_index", {})
        for time_index, cols in secondary_time_indices.items():
            if time_index not in df.columns:
                raise ValueError(
                    'Secondary time index "{}" missing from {} entity'.format(
                        time_index, entity
                    )
                )
            for col in cols:
                if col not in df.columns:
                    raise ValueError(
                        (
                            'Column "{}" associated with secondary time index "{}" '
                            "missing from {} entity"
                        ).format(col, time_index, entity)
                    )


def validate_scada_data(dfs, new_kwargs_mapping=None):
    entity_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
    _validate_data(dfs, "scada", entity_kwargs)
    return entity_kwargs


def validate_pidata_data(dfs, new_kwargs_mapping=None):
    entity_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
    _validate_data(dfs, "pidata", entity_kwargs)
    return entity_kwargs


def validate_vibrations_data(dfs, new_kwargs_mapping=None):
    entities = ["vibrations"]

    pidata_kwargs, scada_kwargs = {}, {}
    if "pidata" in dfs:
        pidata_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
        entities.append("pidata")
    if "scada" in dfs:
        scada_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
        entities.append("scada")

    entity_kwargs = {
        **pidata_kwargs,
        **scada_kwargs,
        **get_mapped_kwargs("vibrations", new_kwargs_mapping),
    }
    _validate_data(dfs, entities, entity_kwargs)
    return entity_kwargs


VALIDATE_DATA_FUNCTIONS = {
    "scada": validate_scada_data,
    "pidata": validate_pidata_data,
    "vibrations": validate_vibrations_data,
}


def _create_entityset(entities, es_type, new_kwargs_mapping=None):
    validate_func = VALIDATE_DATA_FUNCTIONS[es_type]
    es_kwargs = validate_func(entities, new_kwargs_mapping)

    # filter out stated logical types for missing columns
    for entity, df in entities.items():
        es_kwargs[entity]["logical_types"] = {
            col: t
            for col, t in es_kwargs[entity]["logical_types"].items()
            if col in df.columns
        }

    turbines_index = es_kwargs["turbines"]["index"]
    work_orders_index = es_kwargs["work_orders"]["index"]

    relationships = [
        ("turbines", turbines_index, "alarms", turbines_index),
        ("turbines", turbines_index, "stoppages", turbines_index),
        ("turbines", turbines_index, "work_orders", turbines_index),
        ("turbines", turbines_index, es_type, turbines_index),
        ("work_orders", work_orders_index, "notifications", work_orders_index),
    ]

    es = ft.EntitySet()
    es.id = es_type

    for name, df in entities.items():
        es.add_dataframe(dataframe_name=name, dataframe=df, **es_kwargs[name])

    for relationship in relationships:
        parent_df, parent_column, child_df, child_column = relationship
        es.add_relationship(parent_df, parent_column, child_df, child_column)

    return es


CREATE_ENTITYSET_FUNCTIONS = {
    "scada": create_scada_entityset,
    "pidata": create_pidata_entityset,
    "vibrations": create_vibrations_entityset,
}


def get_create_entityset_functions():
    return CREATE_ENTITYSET_FUNCTIONS.copy()


if __name__ == "__main__":
    obj = Zephyr()
    alarms_df = pd.DataFrame(
        {
            "COD_ELEMENT": [0, 0],
            "DAT_START": [
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-03-01 11:12:13"),
            ],
            "DAT_END": [
                pd.Timestamp("2022-01-01 13:00:00"),
                pd.Timestamp("2022-03-02 11:12:13"),
            ],
            "IND_DURATION": [0.5417, 1.0],
            "COD_ALARM": [12345, 98754],
            "COD_ALARM_INT": [12345, 98754],
            "DES_NAME": ["Alarm1", "Alarm2"],
            "DES_TITLE": ["Description of alarm 1", "Description of alarm 2"],
        }
    )
    stoppages_df = pd.DataFrame(
        {
            "COD_ELEMENT": [0, 0],
            "DAT_START": [
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-03-01 11:12:13"),
            ],
            "DAT_END": [
                pd.Timestamp("2022-01-08 11:07:17"),
                pd.Timestamp("2022-03-01 17:00:13"),
            ],
            "DES_WO_NAME": ["stoppage name 1", "stoppage name 2"],
            "DES_COMMENTS": ["description of stoppage 1", "description of stoppage 2"],
            "COD_WO": [12345, 67890],
            "IND_DURATION": [7.4642, 0.2417],
            "IND_LOST_GEN": [45678.0, 123.0],
            "COD_ALARM": [12345, 12345],
            "COD_CAUSE": [32, 48],
            "COD_INCIDENCE": [987654, 123450],
            "COD_ORIGIN": [6, 23],
            "COD_STATUS": ["STOP", "PAUSE"],
            "COD_CODE": ["ABC", "XYZ"],
            "DES_DESCRIPTION": ["Description 1", "Description 2"],
        }
    )
    notifications_df = pd.DataFrame(
        {
            "COD_ELEMENT": [0, 0],
            "COD_ORDER": [12345, 67890],
            "IND_QUANTITY": [1, -20],
            "COD_MATERIAL_SAP": [36052411, 67890],
            "DAT_POSTING": [
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-03-01 00:00:00"),
            ],
            "COD_MAT_DOC": [77889900, 12345690],
            "DES_MEDIUM": [
                "Description of notification 1",
                "Description of notification 2",
            ],
            "COD_NOTIF": [567890123, 32109877],
            "DAT_MALF_START": [
                pd.Timestamp("2021-12-25 18:07:10"),
                pd.Timestamp("2022-02-28 06:04:00"),
            ],
            "DAT_MALF_END": [
                pd.Timestamp("2022-01-08 11:07:17"),
                pd.Timestamp("2022-03-01 17:00:13"),
            ],
            "IND_BREAKDOWN_DUR": [14.1378, 2.4792],
            "FUNCT_LOC_DES": ["location description 1", "location description 2"],
            "COD_ALARM": [12345, 12345],
            "DES_ALARM": ["Alarm description", "Alarm description"],
        }
    )
    work_orders_df = pd.DataFrame(
        {
            "COD_ELEMENT": [0, 0],
            "COD_ORDER": [12345, 67890],
            "DAT_BASIC_START": [
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-03-01 00:00:00"),
            ],
            "DAT_BASIC_END": [
                pd.Timestamp("2022-01-09 00:00:00"),
                pd.Timestamp("2022-03-02 00:00:00"),
            ],
            "COD_EQUIPMENT": [98765, 98765],
            "COD_MAINT_PLANT": ["ABC", "ABC"],
            "COD_MAINT_ACT_TYPE": ["XYZ", "XYZ"],
            "COD_CREATED_BY": ["A1234", "B6789"],
            "COD_ORDER_TYPE": ["A", "B"],
            "DAT_REFERENCE": [
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-03-01 00:00:00"),
            ],
            "DAT_CREATED_ON": [
                pd.Timestamp("2022-03-01 00:00:00"),
                pd.Timestamp("2022-04-18 00:00:00"),
            ],
            "DAT_VALID_END": [pd.NaT, pd.NaT],
            "DAT_VALID_START": [pd.NaT, pd.NaT],
            "COD_SYSTEM_STAT": ["ABC XYZ", "LMN OPQ"],
            "DES_LONG": ["description of work order", "description of work order"],
            "COD_FUNCT_LOC": ["!12345", "?09876"],
            "COD_NOTIF_OBJ": ["00112233", "00998877"],
            "COD_MAINT_ITEM": ["", "019283"],
            "DES_MEDIUM": ["short description", "short description"],
            "DES_FUNCT_LOC": ["XYZ1234", "ABC9876"],
        }
    )
    turbines_df = pd.DataFrame(
        {
            "COD_ELEMENT": [0],
            "TURBINE_PI_ID": ["TA00"],
            "TURBINE_LOCAL_ID": ["A0"],
            "TURBINE_SAP_COD": ["LOC000"],
            "DES_CORE_ELEMENT": ["T00"],
            "SITE": ["LOCATION"],
            "DES_CORE_PLANT": ["LOC"],
            "COD_PLANT_SAP": ["ABC"],
            "PI_COLLECTOR_SITE_NAME": ["LOC0"],
            "PI_LOCAL_SITE_NAME": ["LOC0"],
        }
    )
    pidata_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2022-01-02 13:21:01"),
                pd.Timestamp("2022-03-08 13:21:01"),
            ],
            "COD_ELEMENT": [0, 0],
            "val1": [9872.0, 559.0],
            "val2": [10.0, -7.0],
        }
    )
    obj.create_entityset(
        {
            "alarms": alarms_df,
            "stoppages": stoppages_df,
            "notifications": notifications_df,
            "work_orders": work_orders_df,
            "turbines": turbines_df,
            "pidata": pidata_df,
        },
        "pidata",
    )
    obj.set_labeling_function(name="brake_pad_presence")

    obj.generate_label_times(num_samples=35, gap="20d")
    obj.plot_label_times()

    obj.generate_features(
        target_dataframe_name="turbines",
        cutoff_time_in_index=True,
        agg_primitives=["count", "sum", "max"],
    )

    obj.generate_train_test_split()
    add_primitives_path(
        path="/Users/raymondpan/zephyr/Zephyr-repo/zephyr_ml/primitives/jsons"
    )
    obj.set_pipeline("xgb_classifier")

    obj.fit()

    obj.evaluate()
