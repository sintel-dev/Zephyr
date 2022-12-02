import os

import numpy as np
import pandas as pd

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

    def setup(self):
        self.zephyr = Zephyr('xgb')

    def test_fit(self):
        self.zephyr.fit(self.train, self.train_y)

    def test_predict(self):
        self.zephyr.fit(self.train, self.train_y)

        predicted = self.zephyr.predict(self.test)

        assert self.test_y == predicted

    def test_fit_predict(self):
        predicted = self.zephyr.fit_predict(self.random, self.random_y)

        assert isinstance(predicted, list)

    def test_save_load(self, tmpdir):
        path = os.path.join(tmpdir, 'some/path.pkl')
        self.zephyr.save(path)

        new_zephyr = Zephyr.load(path)
        assert new_zephyr == self.zephyr

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
