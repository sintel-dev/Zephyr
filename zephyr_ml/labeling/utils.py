
import numpy as np
import pandas as pd


def _search_relationship(es, left, right):
    for r in es.relationships:
        if r.parent_name in left:
            if right == r.child_name:
                left_on = r.parent_column.name
                right_on = r.child_column.name

        elif r.child_name in left:
            if right == r.parent_name:
                left_on = r.child_column.name
                right_on = r.parent_column.name

    return left_on, right_on


def denormalize(es, entities):
    """Merge a set of entities into a single dataframe.
    Convert a set of entities from the entityset into a single
    dataframe by repetitively merging the selected entities. The
    merge process is applied sequentially.
    Args:
        entities (list):
            list of strings denoting which entities to merge.
    Returns:
        pandas.DataFrame:
            A single dataframe containing all the information from the
            selected entities.
    """
    k = len(entities)

    # initial entity to start from (should be the target entity)
    first = entities[0]
    previous = [first]
    df = es[first]

    # merge the dataframes to create a single input
    for i in range(1, k):
        right = entities[i]

        left_on, right_on = _search_relationship(es, previous, right)
        df = pd.merge(df, es[right],
                      left_on=left_on, right_on=right_on,
                      how='left', suffixes=('', '_y')).filter(regex='^(?!.*_y)')

        previous.append(right)

    return df


def required_columns(columns):
    """Decorator function for recording required columns for a function."""
    def wrapper(wrapped):
        def func(*args, **kwargs):
            return wrapped(*args, **kwargs)

        func.__required_columns__ = columns
        func.__doc__ = wrapped.__doc__
        func.__name__ = wrapped.__name__
        return func

    return wrapper


def merge_binary_labeling_functions(labeling_functions, and_connected=True):
    """Generates a labeling function from merging multiple binary labeling functions.

    Args:
        labeling_functions (list):
            A list of labeling functions (with df as an input) to merge.
        and_connected (bool):
            If and_connected is True, each individual labeling function criteria must be True
            for the output function to give a positive label. If and_connected is False,
            at least one labeling function criteria has to be met for the output function
            to give a positive label. Default is True.

    Returns:
        function:
            A function that takes in a dataframe, which is derived from the input labeling
            functions.
    """
    def merged_function(df):
        out = and_connected
        for function in labeling_functions:
            if and_connected:
                out &= function(df)
            else:
                out |= function(df)

        return int(out)

    return merged_function


def aggregate_by_column(numerical_column, aggregation):
    """Generates a function for aggregates numerical column values over a data slice.

    Args:
        numerical_column (str):
            Numerical column to aggregate over.
        aggregation (function):
            Aggregation function to apply.

    Returns:
        function:
            The function returns the total numerical column value over the data
            slice as a continuous label.
    """
    def aggregate_function(df):
        """Aggregate function with:
        numerical_column={}
        aggregation={}
        """
        return aggregation(df[numerical_column])

    aggregate_function.__doc__ = aggregate_function.__doc__.format(numerical_column,
                                                                   aggregation.__name__)

    return aggregate_function


def categorical_presence(categorical_column, value):
    """Generates a function that determines if the categorical column has the desired value.

    Args:
        categorical_column (str):
            Categorical column to use values from.
        value (str or int or float):
            Value to compare categorical columns values to.

    Returns:
        function:
            The function returns 1 if categorical column has the desired value,
            0 otherwise.
    """
    def categorical_function(df):
        """Categorical presence function with:
        categorical_column={}
        value={}
        """
        return int(df[categorical_column].isin([value]).sum() > 0)

    categorical_function.__doc__ = categorical_function.__doc__.format(categorical_column, value)
    return categorical_function


def keyword_in_text(keyword, columns=None):
    """Determines presence of keyword in text field data columns.

    Args:
        keyword (str):
            Keyword to search the text columns for.
        columns (list or None):
            List of columns to search through to find keyword. If None, all
            columns are tested. Default is None.

    Returns:
        function:
            The function returns 1 if the keyword is present in any column,
            0 otherwise.
    """
    def keyword_function(df):
        """Keyword function with:
        keyword={}
        columns={}
        """
        mask = np.full(len(df), False)
        for col in columns:
            try:
                mask |= df[col].str.contains(keyword, case=False, na=False)
            except KeyError:
                print("Unable to find column for keyword search")

        return int(mask.sum() != 0)

    keyword_function.__doc__ = keyword_function.__doc__.format(keyword, columns)
    return keyword_function


def greater_than(numerical_column, threshold):
    """Generates a function to see if there are numerical values greater than a threshold.

    Args:
        numerical_column (str):
            Numerical column to use values from.
        threshold (float):
            Threshold for the numerical values used to define the binary labels.

    Returns:
        function:
            The function returns 1 if data contains a value is greater than threshold,
            0 otherwise.
    """
    def numerical_function(df):
        """Numerical presence function with:
        numerical_column={}
        threshold={}
        """
        series = df[numerical_column]
        return int(len(series[series > threshold]) > 0)

    numerical_function.__doc__ = numerical_function.__doc__.format(numerical_column, threshold)
    return numerical_function


def total_duration(start_time, end_time):
    """Generates function for calculating the total duration given start/end time indexes.

    Args:
        start_time (str):
            Name of the start time column.
        end_time (str):
            Name of the end time column.

    Returns:
        function:
            The function returns the total duration in seconds based on the two
            given time endpoints for the data slice.
    """
    def duration_function(df):
        """Duration function with:
        start_time={}
        end_time={}
        """
        return ((df[end_time] - df[start_time]).dt.total_seconds()).sum()

    duration_function.__doc__ = duration_function.__doc__.format(start_time, end_time)
    return duration_function
