import os
import pickle

import numpy as np
import pandas as pd
import pytest

from zephyr_ml.core import Zephyr
import logging


class TestZephyr:

    def base_dfs():
        alarms_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 11:12:13')],
            'DAT_END': [pd.Timestamp('2022-01-01 13:00:00'), pd.Timestamp('2022-03-02 11:12:13')],
            'IND_DURATION': [0.5417, 1.0],
            'COD_ALARM': [12345, 98754],
            'COD_ALARM_INT': [12345, 98754],
            'DES_NAME': ['Alarm1', 'Alarm2'],
            'DES_TITLE': ['Description of alarm 1', 'Description of alarm 2'],
        })
        stoppages_df = pd.DataFrame({
            'COD_ELEMENT': [0, 0],
            'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 11:12:13')],
            'DAT_END': [pd.Timestamp('2022-01-08 11:07:17'), pd.Timestamp('2022-03-01 17:00:13')],
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
            'DAT_POSTING': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 00:00:00')],
            'COD_MAT_DOC': [77889900, 12345690],
            'DES_MEDIUM': ['Description of notification 1', 'Description of notification 2'],
            'COD_NOTIF': [567890123, 32109877],
            'DAT_MALF_START': [pd.Timestamp('2021-12-25 18:07:10'),
                            pd.Timestamp('2022-02-28 06:04:00')],
            'DAT_MALF_END': [pd.Timestamp('2022-01-08 11:07:17'), pd.Timestamp('2022-03-01 17:00:13')],
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
            'time': [pd.Timestamp('2022-01-02 13:21:01'), pd.Timestamp('2022-03-08 13:21:01')],
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
        y_train =X_train['feature 2'].to_list()
        
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
            "create_entityset": {"data_paths": cls.base_dfs(), "es_type": "pidata"},
            "set_labeling_function": {"name": "brake_pad_presence"},
            "generate_label_times": {"num_samples": 10, "gap": "20d"},
            "generate_feature_matrix_and_labels": {"target_dataframe_name": "turbines", "cutoff_time_in_index": True, "agg_primitives": ["count", "sum", "max"], "verbose": True},
            "generate_train_test_split": {},
            "set_and_fit_pipeline": {},
            "evaluate": {}
        }



    def setup_zephyr(self, producer_step_name):
        zephyr = Zephyr()
        step_num = zephyr.producer_to_step_map[producer_step_name]

        for i, (setters, getters) in enumerate(zephyr.step_order):
            if i < step_num:
                setter = setters[0]
                kwargs = self.kwargs[setter.__name__]
                getattr(zephyr, setter.__name__)(**kwargs)
            else:
                break
        return zephyr
    
    def test_initialize_class(self):
        zephyr = self.setup_zephyr(0)
    
    def test_create_entityset(self):
        zephyr = self.setup_zephyr(1)
        es = zephyr.get_entityset()
        assert es is not None
    
    def test_set_labeling_function(self):
        zephyr = self.setup_zephyr(2)
        labeling_fn = es = zephyr.get_labeling_function()
        assert labeling_fn is not None
    
    def test_generate_label_times(self):
        zephyr = self.setup_zephyr(3)
        label_times = zephyr.get_label_times(visualize = False)
        assert label_times is not None
    
    def test_generate_feature_matrix_and_labels(self):
        zephyr = self.setup_zephyr(4)
        feature_matrix_and_labels = zephyr.get_feature_matrix_and_labels()
        assert feature_matrix_and_labels is not None
    
    def test_generate_train_test_split(self):
        zephyr = self.setup_zephyr(5)
        train_test_split = zephyr.get_train_test_split()
        assert train_test_split is not None
    
    def setup_zephyr_with_base_split(self, producer_step_name):
        zephyr = self.setup_zephyr(4)
        zephyr.set_train_test_split(**self.base_train_test_split())
        final_step_num = zephyr.producer_to_step_map[producer_step_name]
        for i in range(4, final_step_num):
            setters, getters = zephyr.step_order[i]
            setter = setters[0]
            kwargs = self.kwargs[setter.__name__]
            getattr(zephyr, setter.__name__)(**kwargs)
        return zephyr
    
    def test_set_train_test_split(self):
        zephyr = self.setup_zephyr_with_base_split(5)
        assert zephyr.get_train_test_split is not None
    
    def test_set_and_fit_pipeline_no_visual(self):
        zephyr = self.setup_zephyr_with_base_split(5)
        output = zephyr.set_and_fit_pipeline()
        assert output is None
        pipeline = zephyr.get_pipeline()
        assert pipeline is not None
        pipeline_hyperparameters = zephyr.get_pipeline_hyperparameters()
        assert pipeline_hyperparameters is not None
    
    def test_set_and_fit_pipeline_visual(self):
        zephyr = self.setup_zephyr_with_base_split(5)
        output = zephyr.set_and_fit_pipeline(visual = True)
        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']
        
        pipeline = zephyr.get_pipeline()
        assert pipeline is not None
        pipeline_hyperparameters = zephyr.get_pipeline_hyperparameters()
        assert pipeline_hyperparameters is not None
    

    def test_predict_no_visual(self):
        zephyr = self.setup_zephyr_with_base_split(6)
        predicted = zephyr.predict()
        _, _, _, test_y = self.base_train_test_split()
        assert test_y == predicted

    def test_predict_visual(self):
        zephyr = self.setup_zephyr_with_base_split(6)
        predicted, output = zephyr.predict(visual = True)

        assert self.test_y == predicted

        # visualization
        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']
    

    def test_evaluate(self):
        zephyr = self.setup_zephyr_with_base_split(6)
        scores = pd.Series(zephyr.evaluate(metrics = ["sklearn.metrics.accuracy_score",
            "sklearn.metrics.precision_score",
            "sklearn.metrics.f1_score",
            "sklearn.metrics.recall_score"]))
        
        expected = pd.Series({
            "sklearn.metrics.accuracy_score": 1.0,
            "sklearn.metrics.precision_score": 1.0,
            "sklearn.metrics.f1_score": 1.0,
            "sklearn.metrics.recall_score": 1.0
        })
        pd.testing.assert_series_equal(expected, scores)
        
        


            
          

    # def setup_method(self):
    #     self.zephyr = Zephyr('xgb_classifier')

    # def test_hyperparameters(self):
    #     hyperparameters = {
    #         "xgboost.XGBClassifier#1": {
    #             "max_depth": 2
    #         },
    #         "zephyr_ml.primitives.postprocessing.FindThreshold#1": {
    #             "metric": "precision"
    #         }
    #     }

    #     zephyr = Zephyr('xgb_classifier', hyperparameters)

    #     assert zephyr._hyperparameters == hyperparameters

    # def test_json(self):
    #     file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     json_zephyr = Zephyr(os.path.join(file, 'zephyr_ml', 'pipelines', 'xgb_classifier.json'))

    #     json_zephyr_hyperparameters = json_zephyr._mlpipeline.get_hyperparameters()
    #     zephyr_hyperparameters = self.zephyr._mlpipeline.get_hyperparameters()
    #     assert json_zephyr_hyperparameters == zephyr_hyperparameters

    # def test_fit(self):
    #     self.zephyr.fit(self.train, self.train_y)

    # def test_fit_visual(self):
    #     output = self.zephyr.fit(self.train, self.train_y, visual=True)

    #     assert isinstance(output, dict)
    #     assert list(output.keys()) == ['threshold', 'scores']

    # def test_fit_no_visual(self):
    #     zephyr = Zephyr(['xgboost.XGBClassifier'])

    #     output = zephyr.fit(self.train, self.train_y, visual=True)
    #     assert output is None

    # def test_predict(self):
    #     self.zephyr.fit(self.train, self.train_y)

    #     predicted = self.zephyr.predict(self.test)

    #     assert self.test_y == predicted

    # def test_predict_visual(self):
    #     self.zephyr.fit(self.train, self.train_y)

    #     predicted, output = self.zephyr.predict(self.test, visual=True)

    #     # predictions
    #     assert self.test_y == predicted

    #     # visualization
    #     assert isinstance(output, dict)
    #     assert list(output.keys()) == ['threshold', 'scores']

    # def test_predict_no_visual(self):
    #     zephyr = Zephyr(['xgboost.XGBClassifier'])

    #     zephyr.fit(self.train, self.train_y)

    #     predicted = zephyr.predict(self.test, visual=True)
    #     assert len(self.test_y) == len(predicted)

    # def test_fit_predict(self):
    #     predicted = self.zephyr.fit_predict(self.random, self.random_y)

    #     assert isinstance(predicted, list)

    # def test_save_load(self, tmpdir):
    #     path = os.path.join(tmpdir, 'some_path.pkl')
    #     self.zephyr.save(path)

    #     new_zephyr = Zephyr.load(path)
    #     assert new_zephyr == self.zephyr

    # def test_load_failed(self, tmpdir):
    #     path = os.path.join(tmpdir, 'some_path.pkl')
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open(path, 'wb') as pickle_file:
    #         pickle.dump("something", pickle_file)

    #     with pytest.raises(ValueError):
    #         Zephyr.load(path)

    # def test_evaluate(self):
    #     self.zephyr.fit(self.test, self.test_y)
    #     scores = self.zephyr.evaluate(X=self.test, y=self.test_y)

    #     expected = pd.Series({
    #         'accuracy': 1.0,
    #         'f1': 1.0,
    #         'recall': 1.0,
    #         'precision': 1.0,
    #     })
    #     pd.testing.assert_series_equal(expected, scores)

    # def test_evaluate_fit(self):
    #     scores = self.zephyr.evaluate(
    #         X=self.test,
    #         y=self.test_y,
    #         fit=True,
    #     )

    #     expected = pd.Series({
    #         'accuracy': 1.0,
    #         'f1': 1.0,
    #         'recall': 1.0,
    #         'precision': 1.0,
    #     })
    #     pd.testing.assert_series_equal(expected, scores)

    # def test_evaluate_previously_fitted_with_fit_true(self):
    #     self.zephyr.fit(self.train, self.train_y)

    #     scores = self.zephyr.evaluate(
    #         X=self.test,
    #         y=self.test_y,
    #         fit=True
    #     )

    #     expected = pd.Series({
    #         'accuracy': 1.0,
    #         'f1': 1.0,
    #         'recall': 1.0,
    #         'precision': 1.0,
    #     })
    #     pd.testing.assert_series_equal(expected, scores)

    # def test_evaluate_train_data(self):
    #     scores = self.zephyr.evaluate(
    #         X=self.test,
    #         y=self.test_y,
    #         fit=True,
    #         train_X=self.train,
    #         train_y=self.train_y
    #     )

    #     expected = pd.Series({
    #         'accuracy': 1.0,
    #         'f1': 1.0,
    #         'recall': 1.0,
    #         'precision': 1.0,
    #     })
    #     pd.testing.assert_series_equal(expected, scores)
