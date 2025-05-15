import copy
import json
import logging
import os
from functools import wraps
from inspect import getfullargspec

import composeml as cp
import featuretools as ft
import numpy as np
import pandas as pd
from mlblocks import MLBlock, MLPipeline
from sklearn.model_selection import train_test_split

from zephyr_ml.entityset import VALIDATE_DATA_FUNCTIONS, _create_entityset
from zephyr_ml.feature_engineering import process_signals
from zephyr_ml.labeling import get_labeling_functions, get_labeling_functions_map

DEFAULT_METRICS = [
    "sklearn.metrics.accuracy_score",
    "sklearn.metrics.precision_score",
    "sklearn.metrics.f1_score",
    "sklearn.metrics.recall_score",
    "zephyr_ml.primitives.postprocessing.confusion_matrix",
    "zephyr_ml.primitives.postprocessing.roc_auc_score_and_curve",
]

LOGGER = logging.getLogger(__name__)


class GuideHandler:

    def __init__(self, producers_and_getters, set_methods):
        self.cur_term = 0
        self.current_step = -1
        self.start_point = -1
        self.producers_and_getters = producers_and_getters
        self.set_methods = set_methods

        self.producer_to_step_map = {}
        self.getter_to_step_map = {}

        self.terms = []
        for idx, (producers, getters) in enumerate(self.producers_and_getters):
            self.terms.append(-1)

            for prod in producers:
                self.producer_to_step_map[prod.__name__] = idx

            for get in getters:
                self.getter_to_step_map[get.__name__] = idx

    def get_necessary_steps(self, actual_next_step):
        step_strs = []
        for step in range(self.current_step, actual_next_step):
            option_strs = []
            for opt in self.producers_and_getters[step][0]:
                option_strs.append(opt.__name__)
            step_strs.append(f"{step}. {' or '.join(option_strs)}")
        return "\n".join(step_strs)

    def get_get_steps_in_between(self, cur_step, next_step):
        step_strs = []
        for step in range(cur_step + 1, next_step):
            step_strs.append(
                f"{step} {self.producers_and_getters[step][1][0]}")
        return step_strs

    def get_last_up_to_date(self, next_step):
        latest_up_to_date = 0
        for step in range(next_step):
            if self.terms[step] == self.cur_term:
                latest_up_to_date = step
        return latest_up_to_date

    def join_steps(self, step_strs):
        return "\n".join(step_strs)

    def get_steps_in_between(self, cur_step, next_step):
        step_strs = []
        for step in range(cur_step + 1, next_step):
            option_strs = []
            for opt in self.producers_and_getters[step][0]:
                option_strs.append(opt.__name__)
            step_strs.append(f"{step}. {' or '.join(option_strs)}")
        return step_strs

    def perform_producer_step(self, zephyr, method, *method_args, **method_kwargs):
        step_num = self.producer_to_step_map[method.__name__]
        res = method(zephyr, *method_args, **method_kwargs)
        self.current_step = step_num
        self.terms[step_num] = self.cur_term
        return res

    def try_log_skipping_steps_warning(self, name, next_step):
        steps_skipped = self.get_steps_in_between(self.current_step, next_step)
        if len(steps_skipped) > 0:
            necc_steps = self.join_steps(steps_skipped)
            LOGGER.warning(
                f"Performing {name}. You are skipping the following steps:\n{necc_steps}")

    def try_log_making_stale_warning(self, name, next_step):
        next_next_step = next_step + 1
        prod_steps = f"step {next_next_step}: \
            {' or '.join(self.producers_and_getters[next_next_step][0])}"
        # add later set methods
        get_steps = self.join_steps(
            self.get_get_steps_in_between(
                next_step, self.current_step + 1))

        LOGGER.warning(f"Performing {name}. You are beginning a new iteration.\
                        Any data returned by the following get methods will be \
                       considered stale:\n{get_steps}. To continue with this \
                        iteration, please perform \n{prod_steps}")

    def log_get_inconsistent_warning(self, name, next_step):
        prod_steps = f"{next_step}. \
            {' or '.join(self.producers_and_getters[next_step][0])}"
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning(f"Unable to perform {name} because {prod_steps} has not \
                        been run yet. Run steps starting at or before \
                        {latest_up_to_date} ")

    def log_get_stale_warning(self, name, next_step):
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning(f"Performing {name}. This data is potentially stale. \
                        Re-run steps starting at or before \
                        {latest_up_to_date} to ensure data is up to date.")

    # tries to perform step if possible -> warns that data might be stale

    def try_perform_forward_producer_step(self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if name in self.set_methods:  # set method will update start point and start new iteration
            self.try_log_skipping_steps_warning(name, next_step)
            self.start_point = next_step
            self.cur_term += 1
        # next_step == 0, set method (already warned), or previous step is up to term
        res = self.perform_producer_step(
            zephyr, method, *method_args, **method_kwargs)
        return res

    # next_step == 0, set method, or previous step is up to term

    def try_perform_backward_producer_step(self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        # starting new iteration
        self.cur_term += 1
        if next_step == 0 or name in self.set_methods:
            self.start_point = next_step
        else:  # key method
            # mark everything from start point to next step as current term
            for i in range(self.start_point, next_step):
                if self.terms[i] != -1:
                    self.terms[i] = self.cur_term

        self.try_log_making_stale_warning(next_step)
        res = self.perform_producer_step(
            zephyr, method, *method_args, **method_kwargs)
        return res

    def try_perform_producer_step(self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if next_step >= self.current_step:
            res = self.try_perform_forward_producer_step(
                zephyr, method, *method_args, **method_kwargs)
            return res
        else:
            res = self.try_perform_backward_producer_step(
                zephyr, method, *method_args, **method_kwargs)
            return res

    # dont update current step or terms

    def try_perform_inconsistent_producer_step(  # add using stale and overwriting
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        # inconsistent forward step: performing key method but previous step is not up to date
        if next_step >= self.current_step and self.terms[next_step-1] != self.cur_term:
            corr_set_method = self.producers_and_getters[next_step][0][1].__name__
            prev_step = next_step-1
            prev_set_method = self.producers_and_getters[prev_step][0][1].__name__
            prev_key_method = self.producers_and_getters[prev_step][0][0].__name__
            LOGGER.warning(f"Unable to perform {name} because you are performing a key method at\
                            step {next_step} but the result of the previous step, \
                            step {prev_step}, is STALE.\
                           If you already have the data for step {next_step}, \
                            you can use the corresponding set method: {corr_set_method}.\
                            Otherwise, please perform step {prev_step} \
                                with {prev_key_method} or {prev_set_method}.")
        # inconsistent backward step: performing set method at nonzero step
        # elif next_step < self.current_step and name in self.set_method:
        #     first_set_method = self.producers_and_getters[0][0][1].__name__
        #     corr_key_method = self.producers_and_getters[next_step][0][0].__name__
        #     LOGGER.warning(f"Unable to perform {name} because you are going backwards \
        #                    and performing step {next_step} with a set method.\
        #                    You can only perform a backwards step with a set \
        #                     method at step 0: {first_set_method}.\
        #                     If you would like to perform step {next_step}, \
        #                         please use the corresponding key method: {corr_key_method}.")
        # inconsistent backward step: performing key method but previous step is not up to date
        elif next_step < self.current_step and self.terms[next_step-1] != self.cur_term:
            prev_step = next_step-1
            prev_key_method = self.producers_and_getters[prev_step][0][0].__name__
            corr_set_method = self.producers_and_getters[next_step][0][1].__name__
            prev_get_method = self.producers_and_getters[prev_step][1][0].__name__
            prev_set_method = self.producers_and_getters[prev_step][0][1].__name__
            LOGGER.warning(f"Unable to perform {name} because you are going \
                           backwards and starting a new iteration by\
                           performing a key method at step {next_step} \
                            but the result of the previous step,\
                            step {prev_step}, is STALE.\
                            If you want to use the STALE result of the PREVIOUS step, \
                            you can call {prev_get_method} to get the data, then\
                            {prev_set_method} to set the data, and then recall this method.\
                            If you want to regenerate the data of the PREVIOUS step, \
                            please call {prev_key_method}, and then recall this method.\
                            If you already have the data for THIS step, you can \
                            call {corr_set_method} to set the data.\
                            ")

    def try_perform_getter_step(self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        # either inconsistent, stale, or up to date
        step_num = self.getter_to_step_map[name]
        step_term = self.terms[step_num]
        if step_term == -1:
            self.log_get_inconsistent_warning(step_num)
        elif step_term == self.cur_term:
            res = method(zephyr, *method_args, **method_kwargs)
            return res
        else:
            self.log_get_stale_warning(step_num)
            res = method(zephyr, *method_args, **method_kwargs)
            return res

    def guide_step(self, zephyr, method, *method_args, **method_kwargs):
        method_name = method.__name__
        if method_name in self.producer_to_step_map:
            # up-todate
            next_step = self.producer_to_step_map[method_name]
            if (next_step == 0 or  # 0 step always valid, starting new iteration
                # set method always valid, but will update start point and start new iteration
                method_name in self.set_methods or
                    # key method valid if previous step is up to date
                    self.terms[next_step-1] == self.cur_term):
                # forward step only valid if set method or key method w/ no skips
                res = self.try_perform_producer_step(
                    zephyr, method, *method_args, **method_kwargs)
                return res
            else:  # stale or inconsistent
                res = self.try_perform_inconsistent_producer_step(
                    zephyr, method, *method_args, **method_kwargs)
                return res
        elif method_name in self.getter_to_step_map:
            res = self.try_perform_getter_step(
                zephyr, method, *method_args, **method_kwargs)
            return res
        else:
            print(f"Method {method_name} does not need to be wrapped")


def guide(method):

    @wraps(method)
    def guided_step(self, *method_args, **method_kwargs):
        return self.guide_handler.guide_step(self, method, *method_args, **method_kwargs)

    return guided_step


class Zephyr:
    """Zephyr Class.

    The Zephyr Class supports all the steps of the predictive engineering workflow
    for wind farm operations data. It manages user state and handles entityset creation, labeling,
    feature engineering, model training and evaluation.
    """

    def __init__(self):
        """Initialize a new Zephyr instance."""
        self._entityset = None

        self._label_times = None
        self._label_times_meta = None

        self._label_col_name = "label"
        self._feature_matrix = None

        self._pipeline = None

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        # tuple of 2 arrays: producers and attributes
        step_order = [
            ([
                self.generate_entityset, self.set_entityset], [
                self.get_entityset]), ([
                    self.generate_label_times, self.set_label_times], [
                    self.get_label_times]), ([
                        self.generate_feature_matrix, self.set_feature_matrix], [
                            self.get_feature_matrix]), ([
                                self.generate_train_test_split, self.set_train_test_split], [
                                    self.get_train_test_split]), ([
                                        self.fit_pipeline, self.set_fitted_pipeline], [
                                            self.get_fitted_pipeline]), ([
                                                self.predict, self.evaluate], [])]
        set_methods = set([self.set_entityset.__name__,
                           self.set_label_times.__name__,
                           self.set_feature_matrix.__name__,
                           self.set_train_test_split.__name__,
                           self.set_fitted_pipeline.__name__])
        self.guide_handler = GuideHandler(step_order, set_methods)

    def GET_ENTITYSET_TYPES(self):
        """Get the supported entityset types and their required dataframes/columns.

        Returns:
            dict: A dictionary mapping entityset types (PI/SCADA/Vibrations) to their
                descriptions and value.
        """
        info_map = {}
        for es_type, val_fn in VALIDATE_DATA_FUNCTIONS.items():
            info_map[es_type] = {"obj": es_type,
                                 "desc": " ".join((val_fn.__doc__.split()))}

        return info_map

    def GET_LABELING_FUNCTIONS(self):
        """Get the available predefined labeling functions.

        Returns:
            dict: A dictionary mapping labeling function names to their
            descriptions and implementations.
        """
        return get_labeling_functions()

    def GET_EVALUATION_METRICS(self):
        """Get the available evaluation metrics.

        Returns:
            dict: A dictionary mapping metric names to their descriptions
             and MLBlock instances.
        """
        info_map = {}
        for metric in DEFAULT_METRICS:
            primitive = self._get_ml_primitive(metric)
            info_map[metric] = {"obj": primitive,
                                "desc": primitive.metadata["description"]}
        return info_map

    @guide
    def generate_entityset(
            self,
            dfs,
            es_type,
            custom_kwargs_mapping=None,
            signal_dataframe_name=None,
            signal_column=None,
            signal_transformations=None,
            signal_aggregations=None,
            signal_window_size=None,
            signal_replace_dataframe=False,
            **sigpro_kwargs):
        """Generate an entityset from input dataframes with optional signal processing.

        Args:
            dfs (dict): Dictionary mapping entity names to pandas DataFrames.
            es_type (str): Type of signal data, either 'SCADA' or 'PI'.
            custom_kwargs_mapping (dict, optional): Custom keyword arguments
                for entityset creation.
            signal_dataframe_name (str, optional): Name of dataframe containing
                signal data to process.
            signal_column (str, optional): Name of column containing signal values to process.
            signal_transformations (list[dict], optional): List of transformation
                primitives to apply.
            signal_aggregations (list[dict], optional): List of aggregation primitives to apply.
            signal_window_size (str, optional): Size of window for signal binning (e.g. '1h').
            signal_replace_dataframe (bool, optional): Whether to replace
                original signal dataframe.
            **sigpro_kwargs: Additional keyword arguments for signal processing.

        Returns:
            featuretools.EntitySet: EntitySet containing the processed data and relationships.
        """
        entityset = _create_entityset(dfs, es_type, custom_kwargs_mapping)

        # perform signal processing
        if signal_dataframe_name is not None and signal_column is not None:
            if signal_transformations is None:
                signal_transformations = []
            if signal_aggregations is None:
                signal_aggregations = []
            process_signals(
                entityset,
                signal_dataframe_name,
                signal_column,
                signal_transformations,
                signal_aggregations,
                signal_window_size,
                signal_replace_dataframe,
                **sigpro_kwargs)

        self._entityset = entityset
        return self._entityset

    @guide
    def set_entityset(self, entityset=None, es_type=None, entityset_path=None,
                      custom_kwargs_mapping=None):
        """Set the entityset for this Zephyr instance.

        Args:
            entityset (featuretools.EntitySet, optional): An existing entityset to use.
            es_type (str, optional): The type of entityset (pi/scada/vibrations).
            entityset_path (str, optional): Path to a saved entityset to load.
            custom_kwargs_mapping (dict, optional): Custom keyword arguments for validation.

        Raises:
            ValueError: If no entityset is provided through any of the parameters.
        """
        if entityset_path is not None:
            entityset = ft.read_entityset(entityset_path)

        if entityset is None:
            raise ValueError(
                "No entityset passed in. Please pass in an entityset object\
                via the entityset parameter or an entityset path via the \
                entityset_path parameter.")

        dfs = entityset.dataframe_dict

        validate_func = VALIDATE_DATA_FUNCTIONS[es_type]
        validate_func(dfs, custom_kwargs_mapping)

        self._entityset = entityset

    @guide
    def get_entityset(self):
        """Get the current entityset.

        Returns:
            featuretools.EntitySet: The current entityset.

        Raises:
            ValueError: If no entityset has been set.
        """
        if self._entityset is None:
            raise ValueError(
                "No entityset has been created or set in this instance.")

        return self._entityset

    @guide
    def generate_label_times(
        self, labeling_fn, num_samples=-1, subset=None, column_map={}, verbose=False, thresh=None,
        window_size=None, minimum_data=None, maximum_data=None, gap=None, drop_empty=True, **kwargs
    ):
        """Generate label times using a labeling function.

        This method applies a labeling function to the entityset to generate labels at specific
        timestamps. The labeling function can be either a predefined one (specified by name) or
        a custom callable.

        Args:
            labeling_fn (callable or str): Either a custom labeling function or the
                name of a predefined function (e.g. 'brake_pad_presence').
                Predefined functions like brake_pad_presence analyze specific patterns
                in the data (e.g. brake pad mentions in stoppage comments) and
                return a tuple containing:
                1) A label generation function that processes data slices
                2) A denormalized dataframe containing the source data
                3) Metadata about the labeling process (e.g. target entity, time index)
            num_samples (int, optional): Number of samples to generate. -1 for all. Defaults to -1.
            subset (int or float, optional): Number or fraction of samples to randomly select.
            column_map (dict, optional): Mapping of column names for the labeling function.
            verbose (bool, optional): Whether to display progress. Defaults to False.
            thresh (float, optional): Threshold for label binarization. If None, tries to
                use threshold value from labeling function metadata, if any.
            window_size (str, optional): Size of the window for label generation (e.g. '1h').
                If None, tries to use window size value from labeling function metadata, if any.
            minimum_data (str, optional): Minimum data required before cutoff time.
            maximum_data (str, optional): Maximum data required after cutoff time.
            gap (str, optional): Minimum gap between consecutive labels.
            drop_empty (bool, optional): Whether to drop windows with no events. Defaults to True.
            **kwargs: Additional arguments passed to the label generation function.

        Returns:
            tuple: (composeml.LabelTimes, dict) The generated label times and metadata.
                Label times contain the generated labels at specific timestamps.
                Metadata contains information about the labeling process.

        Raises:
            ValueError: If labeling_fn is a string but not a recognized predefined function.
            AssertionError: If entityset has not been generated or set or labeling_fn is
                not a string and not callable.
        """
        assert self._entityset is not None, "entityset has not been set"

        if isinstance(labeling_fn, str):  # get predefined labeling function
            labeling_fn_map = get_labeling_functions_map()
            if labeling_fn in labeling_fn_map:
                labeling_fn = labeling_fn_map[labeling_fn]
            else:
                raise ValueError(
                    f"Unrecognized name argument:{labeling_fn}. \
                        Call get_predefined_labeling_functions to \
                            view predefined labeling functions"
                )

        assert callable(labeling_fn), "Labeling function is not callable"

        labeling_function, df, meta = labeling_fn(self._entityset, column_map)

        data = df
        if isinstance(subset, float) or isinstance(subset, int):
            data = data.sample(subset)

        target_entity_index = meta.get("target_entity_index")
        time_index = meta.get("time_index")
        thresh = meta.get("thresh") if thresh is None else thresh
        window_size = meta.get(
            "window_size") if window_size is None else window_size

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
            data.sort_values(time_index), num_samples, minimum_data=minimum_data,
            maximum_data=maximum_data, gap=gap, drop_empty=drop_empty, verbose=verbose, **kwargs
        )
        if thresh is not None:
            label_times = label_times.threshold(thresh)

        self._label_times = label_times
        self._label_col_name = "label"
        self._label_times_meta = meta

        return label_times, meta

    @guide
    def set_label_times(self, label_times, label_col_name, meta=None):
        """Set the label times for this Zephyr instance.

        Args:
            label_times (composeml.LabelTimes): Label times.
            label_col_name (str): Name of the label column.
            meta (dict, optional): Additional metadata about the labels.
        """
        assert (isinstance(label_times, cp.LabelTimes))
        self._label_times = label_times
        self._label_col_name = label_col_name
        self._label_times_meta = meta

    @guide
    def get_label_times(self, visualize=False):
        """Get the current label times.

        Args:
            visualize (bool, optional): Whether to display a distribution plot. Defaults to False.

        Returns:
            tuple: (composeml.LabelTimes, dict) The label times and metadata.
        """
        if visualize:
            cp.label_times.plots.LabelPlots(self._label_times).distribution()
        return self._label_times, self._label_times_meta

    @guide
    def generate_feature_matrix(
            self,
            target_dataframe_name=None,
            instance_ids=None,
            agg_primitives=None,
            trans_primitives=None,
            groupby_trans_primitives=None,
            allowed_paths=None,
            max_depth=2,
            ignore_dataframes=None,
            ignore_columns=None,
            primitive_options=None,
            seed_features=None,
            drop_contains=None,
            drop_exact=None,
            where_primitives=None,
            max_features=-1,
            cutoff_time_in_index=False,
            save_progress=None,
            features_only=False,
            training_window=None,
            approximate=None,
            chunk_size=None,
            n_jobs=1,
            dask_kwargs=None,
            verbose=False,
            return_types=None,
            progress_callback=None,
            include_cutoff_time=True,
            add_interesting_values=False,
            max_interesting_values=5,
            interesting_dataframe_name=None,
            interesting_values=None,
            signal_dataframe_name=None,
            signal_column=None,
            signal_transformations=None,
            signal_aggregations=None,
            signal_window_size=None,
            signal_replace_dataframe=False,
            **sigpro_kwargs):
        """Generate a feature matrix using automated feature engineering.
        Note that this method creates a deepcopy
        of the generated or set entityset within the Zephyr instance
        before performing any signal processing or feature generation.

        Args:
            target_dataframe_name (str, optional): Name of target entity for feature engineering.
            instance_ids (list, optional): List of specific instances to generate features for.
            agg_primitives (list, optional): Aggregation primitives to apply.
            trans_primitives (list, optional): Transform primitives to apply.
            groupby_trans_primitives (list, optional): Groupby transform primitives to apply.
            allowed_paths (list, optional): Allowed entity paths for feature generation.
            max_depth (int, optional): Maximum allowed depth of entity relationships.
                Defaults to 2.
            ignore_dataframes (list, optional): Dataframes to ignore during feature generation.
            ignore_columns (dict, optional): Columns to ignore per dataframe.
            primitive_options (dict, optional): Options for specific primitives.
            seed_features (list, optional): Seed features to begin with.
            drop_contains (list, optional): Drop features containing these substrings.
            drop_exact (list, optional): Drop features exactly matching these names.
            where_primitives (list, optional): Primitives to use in where clauses.
            max_features (int, optional): Maximum number of features to return. -1 for all.
            cutoff_time_in_index (bool, optional): Include cutoff time in the index.
            save_progress (str, optional): Path to save progress.
            features_only (bool, optional): Return only features without calculating values.
            training_window (str, optional): Data window to use for training.
            approximate (str, optional): Approximation method to use.
            chunk_size (int, optional): Size of chunks for parallel processing.
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            dask_kwargs (dict, optional): Arguments for dask computation.
            verbose (bool, optional): Whether to display progress. Defaults to False.
            return_types (list, optional): Types of features to return.
            progress_callback (callable, optional): Callback for progress updates.
            include_cutoff_time (bool, optional): Include cutoff time features. Defaults to True.
            add_interesting_values (bool, optional): Add interesting values. Defaults to False.
            max_interesting_values (int, optional): Maximum interesting values per column.
            interesting_dataframe_name (str, optional): Dataframe for interesting values.
            interesting_values (dict, optional): Pre-defined interesting values.
            signal_dataframe_name (str, optional): Name of dataframe containing signal data.
            signal_column (str, optional): Name of column containing signal values.
            signal_transformations (list, optional): Signal transformations to apply.
            signal_aggregations (list, optional): Signal aggregations to apply.
            signal_window_size (str, optional): Window size for signal processing.
            signal_replace_dataframe (bool, optional): Replace original signal dataframe.
            **sigpro_kwargs: Additional arguments for signal processing.

        Returns:
            tuple: (pd.DataFrame, list, featuretools.EntitySet)
                Feature matrix, feature definitions, and the processed entityset.
        """
        entityset_copy = copy.deepcopy(self._entityset)
        # perform signal processing
        if signal_dataframe_name is not None and signal_column is not None:
            # first make copy of entityset
            if signal_transformations is None:
                signal_transformations = []
            if signal_aggregations is None:
                signal_aggregations = []
            process_signals(
                entityset_copy,
                signal_dataframe_name,
                signal_column,
                signal_transformations,
                signal_aggregations,
                signal_window_size,
                signal_replace_dataframe,
                **sigpro_kwargs)

        # add interesting values for where primitives
        if add_interesting_values:
            entityset_copy.add_interesting_values(
                max_values=max_interesting_values,
                verbose=verbose,
                dataframe_name=interesting_dataframe_name,
                values=interesting_values)

        feature_matrix, features = ft.dfs(
            entityset=entityset_copy, cutoff_time=self._label_times,
            target_dataframe_name=target_dataframe_name,
            instance_ids=instance_ids, agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            groupby_trans_primitives=groupby_trans_primitives,
            allowed_paths=allowed_paths, max_depth=max_depth,
            ignore_dataframes=ignore_dataframes, ignore_columns=ignore_columns,
            primitive_options=primitive_options, seed_features=seed_features,
            drop_contains=drop_contains, drop_exact=drop_exact,
            where_primitives=where_primitives, max_features=max_features,
            cutoff_time_in_index=cutoff_time_in_index,
            save_progress=save_progress, features_only=features_only,
            training_window=training_window, approximate=approximate,
            chunk_size=chunk_size, n_jobs=n_jobs,
            dask_kwargs=dask_kwargs, verbose=verbose,
            return_types=return_types, progress_callback=progress_callback,
            include_cutoff_time=include_cutoff_time
        )
        self._feature_matrix = self._clean_feature_matrix(
            feature_matrix, label_col_name=self._label_col_name)
        self._features = features

        return self._feature_matrix, self._features, entityset_copy

    @guide
    def get_feature_matrix(self):
        """Get the current feature matrix.

        Returns:
            tuple: (pd.DataFrame, str, list) The feature matrix, label column name,
                and feature definitions.
        """
        return self._feature_matrix, self._label_col_name, self._features

    @guide
    def set_feature_matrix(self, feature_matrix, labels=None, label_col_name="label"):
        """Set the feature matrix for this Zephyr instance.

        Args:
            feature_matrix (pd.DataFrame): The feature matrix to use.
            labels (array-like, optional): Labels to add to the feature matrix.
            label_col_name (str, optional): Name of the label column. Defaults to "label".
        """
        assert isinstance(feature_matrix, pd.DataFrame) and (
            labels is not None or
            label_col_name in feature_matrix.columns
        )
        if labels is not None:
            feature_matrix[label_col_name] = labels
        self._feature_matrix = self._clean_feature_matrix(
            feature_matrix, label_col_name=label_col_name
        )
        self._label_col_name = label_col_name

    @guide
    def generate_train_test_split(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=False,
    ):
        """Generate a train-test split of the feature matrix.

        Args:
            test_size (float or int, optional): Proportion or absolute size of test set.
            train_size (float or int, optional): Proportion or absolute size of training set.
            random_state (int, optional): Random seed for reproducibility.
            shuffle (bool, optional): Whether to shuffle before splitting. Defaults to True.
            stratify (bool or list, optional): Whether to maintain label distribution.
                If True, uses labels for stratification. If list, uses those columns.
                Defaults to False.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) The split feature matrices and labels.
        """
        feature_matrix = self._feature_matrix.copy()
        labels = feature_matrix.pop(self._label_col_name)

        if not isinstance(stratify, list):
            if stratify:
                stratify = labels
            else:
                stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            labels,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        return X_train, X_test, y_train, y_test

    @guide
    def set_train_test_split(self, X_train, X_test, y_train, y_test):
        """Set the train-test split for this Zephyr instance.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (array-like): Training labels.
            y_test (array-like): Testing labels.
        """
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

    @guide
    def get_train_test_split(self):
        """Get the current train-test split.

        Returns:
            tuple or None: (X_train, X_test, y_train, y_test) if split exists, None otherwise.
        """
        if (self._X_train is None or self._X_test is None or
                self._y_train is None or self._y_test is None):
            return None
        return self._X_train, self._X_test, self._y_train, self._y_test

    @guide
    def set_fitted_pipeline(self, pipeline):
        """Set a fitted pipeline for this Zephyr instance.

        Args:
            pipeline (MLPipeline): The fitted pipeline to use.
        """
        self._pipeline = pipeline

    @guide
    def fit_pipeline(
            self, pipeline="xgb_classifier", pipeline_hyperparameters=None,
            X=None, y=None, visual=False, **kwargs):
        """Fit a machine learning pipeline.

        Args:
            pipeline (str or dict or MLPipeline, optional): Pipeline to use. Can be:
                - Name of a registered pipeline (default: "xgb_classifier")
                - Path to a JSON pipeline specification
                - Dictionary with pipeline specification
                - MLPipeline instance
            pipeline_hyperparameters (dict, optional): Hyperparameters for the pipeline.
            X (pd.DataFrame, optional): Training features. If None, uses stored training set.
            y (array-like, optional): Training labels. If None, uses stored training labels.
            visual (bool, optional): Whether to return visualization data. Defaults to False.
            **kwargs: Additional arguments passed to the pipeline's fit method.

        Returns:
            dict or None: If visual=True, returns visualization data dictionary.
        """
        self._pipeline = self._get_mlpipeline(
            pipeline, pipeline_hyperparameters)

        if X is None:
            X = self._X_train
        if y is None:
            y = self._y_train

        if visual:
            outputs_spec, visual_names = self._get_outputs_spec(False)
        else:
            outputs_spec = None

        outputs = self._pipeline.fit(X, y, output_=outputs_spec, **kwargs)

        if visual and outputs is not None:
            return dict(zip(visual_names, outputs))

    @guide
    def get_fitted_pipeline(self):
        """Get the current fitted pipeline.

        Returns:
            MLPipeline: The current fitted pipeline.
        """
        return self._pipeline

    @guide
    def predict(self, X=None, visual=False, **kwargs):
        """Make predictions using the fitted pipeline.

        Args:
            X (pd.DataFrame, optional): Features to predict on. If None, uses test set.
            visual (bool, optional): Whether to return visualization data. Defaults to False.
            **kwargs: Additional arguments passed to the pipeline's predict method.

        Returns:
            array-like or tuple: Predictions, and if visual=True, also returns visualization data.
        """
        if X is None:
            X = self._X_test
        if visual:
            outputs_spec, visual_names = self._get_outputs_spec()
        else:
            outputs_spec = "default"

        outputs = self._pipeline.predict(X, output_=outputs_spec, **kwargs)
        if visual and visual_names:
            prediction = outputs[0]
            return prediction, dict(zip(visual_names, outputs[-len(visual_names):]))

        return outputs

    @guide
    def evaluate(
            self, X=None, y=None, metrics=None, global_args=None,
            local_args=None, global_mapping=None, local_mapping=None):
        """Evaluate the fitted pipeline's performance.

        Args:
            X (pd.DataFrame, optional): Features to evaluate on. If None, uses test set.
            y (array-like, optional): True labels. If None, uses test labels.
            metrics (list, optional): Metrics to compute. If None, uses DEFAULT_METRICS.
            global_args (dict, optional): Arguments passed to all metrics.
            local_args (dict, optional): Arguments passed to specific metrics.
            global_mapping (dict, optional): Mapping applied to all metric inputs.
            local_mapping (dict, optional): Mapping applied to specific metric inputs.

        Returns:
            dict: A dictionary mapping metric names to their computed values.
        """
        if X is None:
            X = self._X_test
        if y is None:
            y = self._y_test

        final_context = self._pipeline.predict(X, output_=-1)

        # remap items, if any
        if global_mapping is not None:
            for cur, new in global_mapping.items():
                if cur in final_context:
                    cur_item = final_context.pop(cur)
                    final_context[new] = cur_item

        if metrics is None:
            metrics = DEFAULT_METRICS

        if global_args is None:
            global_args = {}

        if local_args is None:
            local_args = {}

        if local_mapping is None:
            local_mapping = {}

        results = {}
        for metric in metrics:
            try:
                metric_primitive = self._get_ml_primitive(metric)

                if metric in local_mapping:
                    metric_context = {}
                    metric_mapping = local_mapping[metric]
                    for cur, item in final_context.items():
                        new = metric_mapping.get(cur, cur)
                        metric_context[new] = item
                else:
                    metric_context = final_context

                if metric in local_args:
                    metric_args = local_args[metric]
                else:
                    metric_args = {}

                res = metric_primitive.produce(
                    y_true=self._y_test, **metric_context, **metric_args)
                results[metric_primitive.name] = res
            except Exception as e:
                LOGGER.error(
                    f"Unable to run evaluation metric: {metric_primitive.name}",
                    exc_info=e)
        self._results = results
        return results

    def _clean_feature_matrix(self, feature_matrix, label_col_name="label"):
        labels = feature_matrix.pop(label_col_name)

        count_cols = feature_matrix.filter(like="COUNT").columns
        feature_matrix[count_cols] = feature_matrix[count_cols].apply(
            lambda x: x.astype(np.int64)
        )

        string_cols = feature_matrix.select_dtypes(include="category").columns
        feature_matrix = pd.get_dummies(feature_matrix, columns=string_cols)

        feature_matrix[label_col_name] = labels

        return feature_matrix

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
            visual_names = self._pipeline.get_output_names("visual")
            outputs_spec.append("visual")
        except ValueError:
            visual_names = []

        return outputs_spec, visual_names


if __name__ == "__main__":
    obj = Zephyr()
    print(obj.GET_EVALUATION_METRICS())
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

    # obj.create_entityset(
    #     {
    #         "alarms": alarms_df,
    #         "stoppages": stoppages_df,
    #         "notifications": notifications_df,
    #         "work_orders": work_orders_df,
    #         "turbines": turbines_df,
    #         "pidata": pidata_df,
    #     },
    #     "pidata",
    # )

    # obj.set_entityset(entityset_path =
    # "/Users/raymondpan/zephyr/Zephyr-repo/brake_pad_es", es_type = 'scada')

    # obj.set_labeling_function(name="brake_pad_presence")

    # obj.generate_label_times(labeling_fn="brake_pad_presence",
    # num_samples=10, gap="20d")
    # # print(obj.get_label_times())

    # obj.generate_feature_matrix_and_labels(
    #     target_dataframe_name="turbines",
    #     cutoff_time_in_index=True,
    #     agg_primitives=["count", "sum", "max"],
    #     verbose = True
    # )

    # print(obj.get_feature_matrix_and_labels)

    # obj.generate_train_test_split()
    # add_primitives_path(
    #     path="/Users/raymondpan/zephyr/Zephyr-repo/zephyr_ml/primitives/jsons"
    # )
    # obj.set_and_fit_pipeline()

    # obj.evaluate()
