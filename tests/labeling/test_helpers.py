import featuretools as ft
import numpy as np
import pandas as pd

from zephyr_ml.labeling.utils import (
    aggregate_by_column, categorical_presence, denormalize, greater_than, keyword_in_text,
    merge_binary_labeling_functions, total_duration)


def test_aggregate_by_column():
    data = pd.DataFrame({
        'column': [1, 2, 3]
    })
    assert 2 == aggregate_by_column('column', np.mean)(data)


def test_merge_labeling_and_true():
    functions = [
        lambda df: True,
        lambda df: True
    ]
    assert 1 == merge_binary_labeling_functions(functions, and_connected=True)(pd.DataFrame())


def test_merge_labeling_and_false():
    functions = [
        lambda df: True,
        lambda df: False
    ]
    assert 0 == merge_binary_labeling_functions(functions, and_connected=True)(pd.DataFrame())


def test_merge_labeling_or_true():
    functions = [
        lambda df: False,
        lambda df: True
    ]
    assert 1 == merge_binary_labeling_functions(functions, and_connected=False)(pd.DataFrame())


def test_merge_labeling_or_false():
    functions = [
        lambda df: False,
        lambda df: False
    ]
    assert 0 == merge_binary_labeling_functions(functions, and_connected=False)(pd.DataFrame())


def test_categorical_presence_true():
    data = pd.DataFrame({
        'column': ['A', 'B', 'C']
    })
    function = categorical_presence('column', 'A')
    assert 1 == function(data)


def test_categorical_presence_false():
    data = pd.DataFrame({
        'column': ['A', 'B', 'C']
    })
    function = categorical_presence('column', 'D')
    assert 0 == function(data)


def test_keyword_in_text_true():
    data = pd.DataFrame({
        'A': ['this is a comment'],
        'B': ['this is a description']
    })
    function = keyword_in_text('description', columns=['A', 'B'])
    assert 1 == function(data)


def test_keyword_in_text_false():
    data = pd.DataFrame({
        'A': ['this is a comment'],
        'B': ['this is a description']
    })
    function = keyword_in_text('text', columns=['A', 'B'])
    assert 0 == function(data)


def test_keyword_in_unknown_column():
    data = pd.DataFrame({
        'A': ['this is a comment'],
        'B': ['this is a description']
    })
    function = keyword_in_text('text', columns=['A', 'B', 'C'])
    assert 0 == function(data)


def test_greater_than_true():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
    })
    function = greater_than('A', 3)
    assert 1 == function(data)


def test_greater_than_false():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
    })
    function = greater_than('A', 5)
    assert 0 == function(data)


def test_total_duration():
    data = pd.DataFrame({
        'Start': [pd.to_datetime('2010-01-01T10:00'), pd.to_datetime('2010-01-01T14:00')],
        'End': [pd.to_datetime('2010-01-01T11:00'), pd.to_datetime('2010-01-01T16:00')]
    })
    assert 10800 == total_duration('Start', 'End')(data)


def test_total_duration_nan():
    data = pd.DataFrame({
        'Start': [pd.to_datetime('2010-01-01T10:00'), np.nan],
        'End': [pd.to_datetime('2010-01-01T11:00'), pd.to_datetime('2010-01-01T18:00')]
    })
    assert 3600 == total_duration('Start', 'End')(data)


def test_denormalize():
    child_data = pd.DataFrame({
        'child_id': [0, 1, 2],
        'parent_id': ['a', 'a', 'b'],
        'child_value': [100, -5, 25]
    })
    parent_data = pd.DataFrame({
        'parent_id': ['a', 'b'],
        'parent_val': ['x', 'y']
    })

    es = ft.EntitySet(dataframes={'parent': (parent_data, 'parent_id'),
                                  'child': (child_data, 'child_id')},
                      relationships=[('parent', 'parent_id', 'child', 'parent_id')])

    expected = pd.DataFrame({
        'child_id': [0, 1, 2],
        'parent_id': ['a', 'a', 'b'],
        'parent_val': ['x', 'x', 'y'],
        'child_value': [100, -5, 25]
    })
    expected.ww.init()

    actual = denormalize(es, ['child', 'parent'])
    pd.testing.assert_frame_equal(expected, actual, check_like=True)

    actual2 = denormalize(es, ['parent', 'child'])
    pd.testing.assert_frame_equal(expected, actual2, check_like=True)
