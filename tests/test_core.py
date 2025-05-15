import numpy as np
import pandas as pd
from mlblocks import MLBlock

from zephyr_ml.core import DEFAULT_METRICS, Zephyr


class TestZephyr:

    @staticmethod
    def base_dfs():
        alarms_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'),
                          pd.Timestamp('2022-03-01 11:12:13')],
            'DAT_END': [pd.Timestamp('2022-01-01 13:00:00'),
                        pd.Timestamp('2022-03-02 11:12:13')],
            'IND_DURATION': [0.5417, 1.0],
            'COD_ALARM': [12345, 98754],
            'COD_ALARM_INT': [12345, 98754],
            'DES_NAME': ['Alarm1', 'Alarm2'],
            'DES_TITLE': ['Description of alarm 1', 'Description of alarm 2'],
        })
        stoppages_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'),
                          pd.Timestamp('2022-03-01 11:12:13')],
            'DAT_END': [pd.Timestamp('2022-01-08 11:07:17'),
                        pd.Timestamp('2022-03-01 17:00:13')],
            'DES_WO_NAME': ['stoppage name 1', 'stoppage name 2'],
            'DES_COMMENTS': ['description of stoppage 1', 'description of stoppage 2'],
            'COD_WO': [12345, 67890],
            'IND_DURATION': [7.4642, 0.2417],
            'IND_LOST_GEN': [45678.0, 123.0],
            'COD_ALARM': [12345, 12345],
            'COD_CAUSE': [32, 48],
            'COD_INCIDENCE': [987654, 123450],
            'COD_ORIGIN': [6, 23],
            'COD_STATUS': ['STOP', 'PAUSE'],
            'COD_CODE': ['ABC', 'XYZ'],
            'DES_DESCRIPTION': ['Description 1', 'Description 2']
        })
        notifications_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'COD_ORDER': [12345, 67890],
            'IND_QUANTITY': [1, -20],
            'COD_MATERIAL_SAP': [36052411, 67890],
            'DAT_POSTING': [pd.Timestamp('2022-01-01 00:00:00'),
                            pd.Timestamp('2022-03-01 00:00:00')],
            'COD_MAT_DOC': [77889900, 12345690],
            'DES_MEDIUM': ['Description of notification 1', 'Description of notification 2'],
            'COD_NOTIF': [567890123, 32109877],
            'DAT_MALF_START': [pd.Timestamp('2021-12-25 18:07:10'),
                               pd.Timestamp('2022-02-28 06:04:00')],
            'DAT_MALF_END': [pd.Timestamp('2022-01-08 11:07:17'),
                             pd.Timestamp('2022-03-01 17:00:13')],
            'IND_BREAKDOWN_DUR': [14.1378, 2.4792],
            'FUNCT_LOC_DES': ['location description 1', 'location description 2'],
            'COD_ALARM': [12345, 12345],
            'DES_ALARM': ['Alarm description', 'Alarm description'],
        })
        work_orders_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'COD_ORDER': [12345, 67890],
            'DAT_BASIC_START': [pd.Timestamp('2022-01-01 00:00:00'),
                                pd.Timestamp('2022-03-01 00:00:00')],
            'DAT_BASIC_END': [pd.Timestamp('2022-01-09 00:00:00'),
                              pd.Timestamp('2022-03-02 00:00:00')],
            'COD_EQUIPMENT': [98765, 98765],
            'COD_MAINT_PLANT': ['ABC', 'ABC'],
            'COD_MAINT_ACT_TYPE': ['XYZ', 'XYZ'],
            'COD_CREATED_BY': ['A1234', 'B6789'],
            'COD_ORDER_TYPE': ['A', 'B'],
            'DAT_REFERENCE': [pd.Timestamp('2022-01-01 00:00:00'),
                              pd.Timestamp('2022-03-01 00:00:00')],
            'DAT_CREATED_ON': [pd.Timestamp('2022-03-01 00:00:00'),
                               pd.Timestamp('2022-04-18 00:00:00')],
            'DAT_VALID_END': [pd.NaT, pd.NaT],
            'DAT_VALID_START': [pd.NaT, pd.NaT],
            'COD_SYSTEM_STAT': ['ABC XYZ', 'LMN OPQ'],
            'DES_LONG': ['description of work order', 'description of work order'],
            'COD_FUNCT_LOC': ['!12345', '?09876'],
            'COD_NOTIF_OBJ': ['00112233', '00998877'],
            'COD_MAINT_ITEM': ['', '019283'],
            'DES_MEDIUM': ['short description', 'short description'],
            'DES_FUNCT_LOC': ['XYZ1234', 'ABC9876'],
        })
        turbines_df = pd.DataFrame({
            'COD_ELEMENT': [0],
            'TURBINE_PI_ID': ['TA00'],
            'TURBINE_LOCAL_ID': ['A0'],
            'TURBINE_SAP_COD': ['LOC000'],
            'DES_CORE_ELEMENT': ['T00'],
            'SITE': ['LOCATION'],
            'DES_CORE_PLANT': ['LOC'],
            'COD_PLANT_SAP': ['ABC'],
            'PI_COLLECTOR_SITE_NAME': ['LOC0'],
            'PI_LOCAL_SITE_NAME': ['LOC0']
        })
        pidata_df = pd.DataFrame({
            'time': [pd.Timestamp('2022-01-02 13:21:01'),
                     pd.Timestamp('2022-03-08 13:21:01')],
            'COD_ELEMENT': [0, 0],
            'val1': [9872.0, 559.0],
            'val2': [10.0, -7.0]
        })
        return {
            'alarms': alarms_df,
            'stoppages': stoppages_df,
            'notifications': notifications_df,
            'work_orders': work_orders_df,
            'turbines': turbines_df,
            "pidata": pidata_df
        }

    def base_train_test_split(self):
        X_train = pd.DataFrame({
            'feature 1': np.random.random(300),
            'feature 2': [0] * 150 + [1] * 150,
        })
        y_train = X_train['feature 2'].to_list()
        X_test = pd.DataFrame({
            'feature 1': np.random.random((100)),
            'feature 2': [0] * 25 + [1] * 50 + [0] * 25,
        })
        y_test = X_test['feature 2'].to_list()
        return X_train, X_test, y_train, y_test

    @classmethod
    def setup_class(cls):
        cls.train = pd.DataFrame({
            'feature 1': np.random.random(300),
            'feature 2': [0] * 150 + [1] * 150,
        })
        cls.train_y = cls.train['feature 2'].to_list()
        cls.test = pd.DataFrame({
            'feature 1': np.random.random((100)),
            'feature 2': [0] * 25 + [1] * 50 + [0] * 25,
        })
        cls.test_y = cls.test['feature 2'].to_list()
        cls.random = pd.DataFrame({
            'feature 1': list(range(100)),
            'feature 2': np.random.random(100),
            'feature 3': np.random.random(100),
        })
        cls.random_y = [1 if x > 0.5 else 0 for x in np.random.random(100)]
        cls.kwargs = {
            "generate_entityset": {
                "dfs": TestZephyr.base_dfs(),
                "es_type": "pidata"},
            "generate_label_times": {
                "labeling_fn": "brake_pad_presence",
                "num_samples": 10,
                "gap": "20d"},
            "generate_feature_matrix": {
                "target_dataframe_name": "turbines",
                "cutoff_time_in_index": True,
                "agg_primitives": [
                    "count",
                    "sum",
                    "max"],
                "verbose": True},
            "generate_train_test_split": {},
            "fit_pipeline": {},
            "evaluate": {}}

    def test_initialize_class(self):
        _ = Zephyr()

    def test_generate_entityset(self):
        zephyr = Zephyr()
        zephyr.generate_entityset(
            **self.__class__.kwargs["generate_entityset"])
        es = zephyr.get_entityset()
        assert es is not None
        assert es.id == 'pidata'

    def test_generate_label_times(self):
        zephyr = Zephyr()
        zephyr.generate_entityset(
            **self.__class__.kwargs["generate_entityset"])
        zephyr.generate_label_times(
            **self.__class__.kwargs["generate_label_times"])
        label_times = zephyr.get_label_times(visualize=False)
        assert label_times is not None

    def test_generate_feature_matrix_and_labels(self):
        zephyr = Zephyr()
        zephyr.generate_entityset(
            **self.__class__.kwargs["generate_entityset"])
        zephyr.generate_label_times(
            **self.__class__.kwargs["generate_label_times"])
        zephyr.generate_feature_matrix(
            **self.__class__.kwargs["generate_feature_matrix"])
        feature_matrix, label_col_name, features = zephyr.get_feature_matrix()
        assert feature_matrix is not None
        assert label_col_name in feature_matrix.columns
        assert features is not None

    def test_generate_train_test_split(self):
        zephyr = Zephyr()
        zephyr.generate_entityset(
            **self.__class__.kwargs["generate_entityset"])
        zephyr.generate_label_times(
            **self.__class__.kwargs["generate_label_times"])
        zephyr.generate_feature_matrix(
            **self.__class__.kwargs["generate_feature_matrix"])
        zephyr.generate_train_test_split(
            **self.__class__.kwargs["generate_train_test_split"])
        train_test_split = zephyr.get_train_test_split()
        assert train_test_split is not None
        X_train, X_test, y_train, y_test = train_test_split
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_set_train_test_split(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        train_test_split = zephyr.get_train_test_split()
        assert train_test_split is not None
        X_train, X_test, y_train, y_test = train_test_split
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, list)
        assert isinstance(y_test, list)

    def test_fit_pipeline_no_visual(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        output = zephyr.fit_pipeline(**self.__class__.kwargs["fit_pipeline"])
        assert output is None
        pipeline = zephyr.get_fitted_pipeline()
        assert pipeline is not None

    def test_fit_pipeline_visual(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        output = zephyr.fit_pipeline(
            visual=True, **self.__class__.kwargs["fit_pipeline"])
        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']
        pipeline = zephyr.get_fitted_pipeline()
        assert pipeline is not None

    def test_predict_no_visual(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        zephyr.fit_pipeline(**self.__class__.kwargs["fit_pipeline"])
        predicted = zephyr.predict()
        _, _, _, test_y = self.base_train_test_split()
        assert predicted == test_y

    def test_predict_visual(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        zephyr.fit_pipeline(**self.__class__.kwargs["fit_pipeline"])
        predicted, output = zephyr.predict(visual=True)
        assert isinstance(predicted, list)
        assert len(predicted) == len(self.test_y)
        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']

    def test_evaluate(self):
        zephyr = Zephyr()
        zephyr.set_train_test_split(*self.base_train_test_split())
        zephyr.fit_pipeline(**self.__class__.kwargs["fit_pipeline"])
        scores = zephyr.evaluate(metrics=[
            "sklearn.metrics.accuracy_score",
            "sklearn.metrics.precision_score",
            "sklearn.metrics.f1_score",
            "sklearn.metrics.recall_score"
        ])
        assert isinstance(scores, dict)
        assert all(metric in scores for metric in [
            "sklearn.metrics.accuracy_score",
            "sklearn.metrics.precision_score",
            "sklearn.metrics.f1_score",
            "sklearn.metrics.recall_score"
        ])

    def test_get_entityset_types(self):
        zephyr = Zephyr()
        entityset_types = zephyr.GET_ENTITYSET_TYPES()
        assert isinstance(entityset_types, dict)
        assert "pidata" in entityset_types
        assert "scada" in entityset_types
        assert "vibrations" in entityset_types
        for es_type, info in entityset_types.items():
            assert isinstance(info, dict)
            assert "obj" in info
            assert "desc" in info
            assert isinstance(info["obj"], str)
            assert isinstance(info["desc"], str)

    def test_get_labeling_functions(self):
        zephyr = Zephyr()
        labeling_functions = zephyr.GET_LABELING_FUNCTIONS()
        assert isinstance(labeling_functions, dict)
        assert "brake_pad_presence" in labeling_functions
        for func_name, info in labeling_functions.items():
            assert isinstance(info, dict)
            assert "obj" in info
            assert "desc" in info
            assert callable(info["obj"])
            assert isinstance(info["desc"], str)

    def test_get_evaluation_metrics(self):
        zephyr = Zephyr()
        evaluation_metrics = zephyr.GET_EVALUATION_METRICS()
        assert isinstance(evaluation_metrics, dict)
        expected_metrics = DEFAULT_METRICS
        for metric in expected_metrics:
            assert metric in evaluation_metrics
        for metric_name, info in evaluation_metrics.items():
            assert isinstance(info, dict)
            assert "obj" in info
            assert "desc" in info
            assert isinstance(info["obj"], MLBlock)
            assert hasattr(info["obj"], "metadata")
            assert isinstance(info["desc"], str)
