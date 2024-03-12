from zephyr_ml.labeling import utils
from zephyr_ml.labeling.data_labeler import DataLabeler
from zephyr_ml.labeling.labeling_functions import (
    brake_pad_presence, converter_replacement_presence, gearbox_replace_presence, total_power_loss)

LABELING_FUNCTIONS = [
    brake_pad_presence,
    converter_replacement_presence,
    gearbox_replace_presence,
    total_power_loss
]
UTIL_FUNCTIONS = [
    utils.aggregate_by_column,
    utils.categorical_presence,
    utils.greater_than,
    utils.keyword_in_text,
    utils.merge_binary_labeling_functions,
    utils.total_duration,
]


def get_labeling_functions():
    functions = {}
    for function in LABELING_FUNCTIONS:
        name = function.__name__
        functions[name] = function.__doc__.split('\n')[0]

    return functions


def get_helper_functions():
    functions = {}
    for function in UTIL_FUNCTIONS:
        name = function.__name__
        functions[name] = function.__doc__.split('\n')[0]

    return functions


def get_util_functions():
    return get_helper_functions()
