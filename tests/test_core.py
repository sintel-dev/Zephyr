import os
import pickle

import numpy as np
import pandas as pd
import pytest

from zephyr_ml.core import Zephyr


class TestZephyr:

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

    def setup_method(self):
        self.zephyr = Zephyr('xgb_classifier')

    def test_hyperparameters(self):
        hyperparameters = {
            "xgboost.XGBClassifier#1": {
                "max_depth": 2
            },
            "zephyr_ml.primitives.postprocessing.FindThreshold#1": {
                "metric": "precision"
            }
        }

        zephyr = Zephyr('xgb_classifier', hyperparameters)

        assert zephyr._hyperparameters == hyperparameters

    def test_json(self):
        file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_zephyr = Zephyr(os.path.join(file, 'zephyr_ml', 'pipelines', 'xgb_classifier.json'))

        json_zephyr_hyperparameters = json_zephyr._mlpipeline.get_hyperparameters()
        zephyr_hyperparameters = self.zephyr._mlpipeline.get_hyperparameters()
        assert json_zephyr_hyperparameters == zephyr_hyperparameters

    def test_fit(self):
        self.zephyr.fit(self.train, self.train_y)

    def test_fit_visual(self):
        output = self.zephyr.fit(self.train, self.train_y, visual=True)

        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']

    def test_fit_no_visual(self):
        zephyr = Zephyr(['xgboost.XGBClassifier'])

        output = zephyr.fit(self.train, self.train_y, visual=True)
        assert output is None

    def test_predict(self):
        self.zephyr.fit(self.train, self.train_y)

        predicted = self.zephyr.predict(self.test)

        assert self.test_y == predicted

    def test_predict_visual(self):
        self.zephyr.fit(self.train, self.train_y)

        predicted, output = self.zephyr.predict(self.test, visual=True)

        # predictions
        assert self.test_y == predicted

        # visualization
        assert isinstance(output, dict)
        assert list(output.keys()) == ['threshold', 'scores']

    def test_predict_no_visual(self):
        zephyr = Zephyr(['xgboost.XGBClassifier'])

        zephyr.fit(self.train, self.train_y)

        predicted = zephyr.predict(self.test, visual=True)
        assert len(self.test_y) == len(predicted)

    def test_fit_predict(self):
        predicted = self.zephyr.fit_predict(self.random, self.random_y)

        assert isinstance(predicted, list)

    def test_save_load(self, tmpdir):
        path = os.path.join(tmpdir, 'some_path.pkl')
        self.zephyr.save(path)

        new_zephyr = Zephyr.load(path)
        assert new_zephyr == self.zephyr

    def test_load_failed(self, tmpdir):
        path = os.path.join(tmpdir, 'some_path.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as pickle_file:
            pickle.dump("something", pickle_file)

        with pytest.raises(ValueError):
            Zephyr.load(path)

    def test_evaluate(self):
        self.zephyr.fit(self.test, self.test_y)
        scores = self.zephyr.evaluate(X=self.test, y=self.test_y)

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)

    def test_evaluate_fit(self):
        scores = self.zephyr.evaluate(
            X=self.test,
            y=self.test_y,
            fit=True,
        )

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)

    def test_evaluate_previously_fitted_with_fit_true(self):
        self.zephyr.fit(self.train, self.train_y)

        scores = self.zephyr.evaluate(
            X=self.test,
            y=self.test_y,
            fit=True
        )

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)

    def test_evaluate_train_data(self):
        scores = self.zephyr.evaluate(
            X=self.test,
            y=self.test_y,
            fit=True,
            train_X=self.train,
            train_y=self.train_y
        )

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)
