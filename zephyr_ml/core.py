from zephyr_ml.entityset import _create_entityset, VALIDATE_DATA_FUNCTIONS
from zephyr_ml.labeling import get_labeling_functions, get_labeling_functions_map, LABELING_FUNCTIONS
# from zephyr_ml.entityset import _create_entityset, VALIDATE_DATA_FUNCTIONS
from zephyr_ml.labeling import (
    get_labeling_functions,
    get_labeling_functions_map,
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
import logging
import matplotlib.pyplot as plt
from functools import wraps

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
            step_strs.append(f"{step} {self.producers_and_getters[step][1][0]}")
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
        for step in range(cur_step+1, next_step):
            option_strs = []
            for opt in self.producers_and_getters[step][0]:
                option_strs.append(opt.__name__)
            step_strs.append(f"{step}. {' or '.join(option_strs)}")
        return step_strs
    
    def perform_producer_step(self, method, *method_args, **method_kwargs):
        step_num = self.producer_to_step_map[method.__name__]
        res = method(*method_args, **method_kwargs)
        self.current_step = step_num
        self.terms[step_num] = self.cur_term
        return res
    
    
    def try_log_skipping_steps_warning(self, name, next_step):
        steps_skipped = self.get_steps_in_between(self.current_step, next_step)
        if len(steps_skipped) > 0:
            necc_steps = self.join_steps(steps_skipped)
            LOGGER.warning(f"Performing {name}. You are skipping the following steps:\n{necc_steps}")
           

    def try_log_using_stale_warning(self, name, next_step):
        latest_up_to_date = self.get_last_up_to_date(next_step)
        steps_needed = self.get_steps_in_between(latest_up_to_date-1, next_step)
        if len(steps_needed) >0:
            necc_steps = self.join_steps(steps_needed)
            LOGGER.warning(f"Performing {name}. You are in a stale state and \
                        using potentially stale data to perform this step. \
                        Re-run the following steps to return to a present state:\n: \
                        {steps_needed}")
        

    def try_log_making_stale_warning(self, name, next_step):
        next_next_step = next_step + 1
        prod_steps = f"{next_next_step}. {" or ".join(self.producers_and_getters[next_next_step][0])}"
        # add later set methods
        get_steps = self.join_steps(self.get_get_steps_in_between(next_step, self.current_step + 1))


        LOGGER.warning(f"Performing {name}. You are beginning a new iteration. Any data returned \
                       by the following get methods will be considered stale:\n{get_steps}. To continue with this iteration, please perform:\n{prod_steps}")

    # stale must be before b/c user must have regressed with progress that contains skips
    # return set method, and next possible up to date key method
    def try_log_inconsistent_warning(self, name, next_step):
        set_method_str= f"{self.producers_and_getters[next_step][0][1].__name__}"
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning(f"Unable to perform {name} because some steps have been skipped. \
                       You can call the corresponding set method: {set_method_str} or re run steps \
                        starting at or before {latest_up_to_date}")
        
    def log_get_inconsistent_warning(self, name, next_step):
        prod_steps = f"{next_step}. {" or ".join(self.producers_and_getters[next_step][0])}"
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning(f"Unable to perform {name} because {prod_steps} has not been run yet. Run steps starting at or before {latest_up_to_date} ")
        

    def log_get_stale_warning(self, name, next_step):
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning(f"Performing {name}. This data is potentially stale. \
                       Re-run steps starting at or before {latest_up_to_date} to ensure data is up to date.")
        

    # tries to perform step if possible -> warns that data might be stale    
    def try_perform_forward_producer_step(self, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if name in self.set_methods:
            self.try_log_skipping_steps_warning(name, next_step)
        # next_step == 0, set method (already warned), or previous step is up to term
        res = self.perform_producer_step(method, *method_args, **method_kwargs)
        return res
    

    # next_step == 0, set method, or previous step is up to term
    def try_perform_backward_producer_step(self, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        self.try_log_making_stale_warning(next_step)
        self.cur_term +=1
        for i in range(0, next_step):
            if self.terms[i] != -1:
                self.terms[i] = self.cur_term
        res = self.perform_producer_step(method, *method_args, **method_kwargs)
        return res


    def try_perform_producer_step(self, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if next_step >= self.current_step:
            res = self.try_perform_forward_producer_step(method, *method_args, **method_kwargs)
            return res
        else:
            res = self.try_perform_backward_producer_step(method, *method_args, **method_kwargs)
            return res


    # dont update current step or terms
    def try_perform_stale_or_inconsistent_producer_step(self, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if self.terms[next_step-1] == -1: #inconsistent
            self.try_log_inconsistent_warning(name, next_step)
        else:
            self.try_log_using_stale_warning(name, next_step)
            res = self.perform_producer_step(method, *method_args, **method_kwargs)
            return res


    

    

    def try_perform_getter_step(self, method, *method_args, **method_kwargs):
        name = method.__name__
        # either inconsistent, stale, or up to date
        step_num = self.getter_to_step_map[name]
        step_term = self.terms[step_num]
        if step_term == -1:
            self.log_get_inconsistent_warning(step_num)
        elif step_term == self.cur_term:
            res = method(*method_args, **method_kwargs)
            return res
        else:
            self.log_get_stale_warning(step_num)
            res = method(*method_args, **method_kwargs)
            return res




    

    def guide_step(self, method, *method_args, **method_kwargs):
        method_name = method.__name__
        if method_name in self.producer_to_step_map:
            #up-todate
            next_step = self.producer_to_step_map[method_name]
            if method_name in self.set_methods or next_step == 0 or self.terms[next_step-1] == self.cur_term:
                res = self.try_perform_producer_step(method, *method_args, **method_kwargs)
                return res
            else: #stale or inconsistent
                res = self.try_perform_stale_or_inconsistent_producer_step(method, *method_args, **method_kwargs)
                return res
        elif method_name in self.getter_to_step_map:
            res = self.try_perform_getter_step(method, *method_args, **method_kwargs)
            return res
        else:
            print(f"Method {method_name} does not need to be wrapped")






        

    




def guide(method):

    @wraps(method)
    def guided_step(self, *method_args, **method_kwargs):
        return self.guide_handler.guide_step(method, *method_args, **method_kwargs)

    return guided_step

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
        self.is_fitted = None
        self.results = None


        
        self.current_step = -1
        # tuple of 2 arrays: producers and attributes
        self.step_order = [
            ([self.generate_entityset, self.set_entityset], [self.get_entityset]),
            # ([self.set_labeling_function], [self.get_labeling_function]),
            ([self.generate_label_times, self.set_label_times], [self.get_label_times]),
            ([self.generate_feature_matrix_and_labels, self.set_feature_matrix_and_labels], [self.get_feature_matrix_and_labels]),
            ([self.generate_train_test_split, self.set_train_test_split], [self.get_train_test_split]),
            ([self.fit_pipeline, self.set_fitted_pipeline], [self.get_fitted_pipeline]),
            ([self.predict, self.evaluate], [])
        ]
        self.set_methods = set([self.set_entityset.__name__, self.set_label_times.__name__, self.set_feature_matrix_and_labels.__name__, self.set_train_test_split.__name__, self.set_fitted_pipeline.__name__])
        self.guide_handler = GuideHandler(self.step_order, self.set_methods) 
    

            

    def GET_ENTITYSET_TYPES(self):
        """
        Returns the supported entityset types (PI/SCADA/Vibrations) and the required dataframes and their columns
        """
        return VALIDATE_DATA_FUNCTIONS.keys()

    @guide
    def generate_entityset(self, dfs, es_type, custom_kwargs_mapping=None):
        """
        Generate an entityset

        Args:
        dfs ( dict ): Dictionary mapping entity names to the pandas
        dataframe for that that entity
        es_type (str): type of signal data , either SCADA or PI
        custom_kwargs_mapping ( dict ): Updated keyword arguments to be used
        during entityset creation
        Returns:
        featuretools.EntitySet that contains the data passed in and
        their relationships
        """
        entityset = _create_entityset(dfs, es_type, custom_kwargs_mapping)
        self.entityset = entityset
        return self.entityset

    @guide
    def set_entityset(self, entityset=None, es_type=None, entityset_path = None, custom_kwargs_mapping=None):
        if entityset_path is not None:
            entityset = ft.read_entityset(entityset_path)

        if entityset is None:
            raise ValueError("No entityset passed in. Please pass in an entityset object via the entityest parameter or an entityset path via the entityset_path parameter.")
        
        dfs = entityset.dataframe_dict

        validate_func = VALIDATE_DATA_FUNCTIONS[es_type]
        validate_func(dfs, custom_kwargs_mapping)

        self.entityset = entityset

    @guide
    def get_entityset(self):
        if self.entityset is None:
            raise ValueError("No entityset has been created or set in this instance.")

        return self.entityset
    

    def GET_LABELING_FUNCTIONS(self):
        return get_labeling_functions()

    # @guide
    # def set_labeling_function(self, name=None, func=None):
    #     if name is not None:
    #         labeling_fn_map = get_labeling_functions_map()
    #         if name in labeling_fn_map:
    #             self.labeling_function = labeling_fn_map[name]
    #             return
    #         else:
    #             raise ValueError(
    #                 f"Unrecognized name argument:{name}. Call get_predefined_labeling_functions to view predefined labeling functions"
    #             )
    #     elif func is not None:
    #         if callable(func):
    #             self.labeling_function = func
    #             return
    #         else:
    #             raise ValueError(f"Custom function is not callable")
    #     raise ValueError("No labeling function given.")
    
    # @guide
    # def get_labeling_function(self):
    #     return self.labeling_function
    
    @guide
    def generate_label_times(
        self, labeling_fn, num_samples=-1, subset=None, column_map={}, verbose=False, **kwargs
    ):
        assert self.entityset is not None, "entityset has not been set"
        
        if isinstance(labeling_fn, str): # get predefined labeling function
            labeling_fn_map = get_labeling_functions_map()
            if labeling_fn in labeling_fn_map:
                labeling_fn = labeling_fn_map[labeling_fn]
            else:
                raise ValueError(
                    f"Unrecognized name argument:{labeling_fn}. Call get_predefined_labeling_functions to view predefined labeling functions"
                )


        assert callable(labeling_fn), "Labeling function is not callable"
            

        labeling_function, df, meta = labeling_fn(self.entityset, column_map)

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
    
    @guide
    def set_label_times(self, label_times):
        assert(isinstance(label_times, cp.LabelTimes))
        self.label_times = label_times

    @guide
    def get_label_times(self, visualize = True):
        if visualize:
            cp.label_times.plots.LabelPlots(self.label_times).distribution()
        return self.label_times

    @guide
    def generate_feature_matrix_and_labels(self, **kwargs):
        feature_matrix, features = ft.dfs(
            entityset=self.entityset, cutoff_time=self.label_times, **kwargs
        )
        self.feature_matrix_and_labels = self._clean_feature_matrix(feature_matrix)
        self.features = features
        return self.feature_matrix_and_labels, features

    @guide
    def get_feature_matrix_and_labels(self):
        return self.feature_matrix_and_labels


    @guide
    def set_feature_matrix_and_labels(self, feature_matrix, label_col_name="label"):
        assert label_col_name in feature_matrix.columns
        self.feature_matrix_and_labels = self._clean_feature_matrix(
            feature_matrix, label_col_name=label_col_name
        )

    @guide
    def generate_train_test_split(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=False,
    ):
        feature_matrix, labels = self.feature_matrix_and_labels
        
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
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return

    @guide
    def set_train_test_split(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @guide
    def get_train_test_split(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            return None
        return self.X_train, self.X_test, self.y_train, self.y_test

    
    
    @guide
    def set_fitted_pipeline(self, pipeline):
        self.pipeline = pipeline
    
    @guide
    def fit_pipeline(
        self, pipeline = "xgb_classifier", pipeline_hyperparameters=None, X=None, y=None, visual=False, **kwargs
    ):  # kwargs indicate the parameters of the current pipeline
        self.pipeline = self._get_mlpipeline(pipeline, pipeline_hyperparameters)

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
        
    @guide
    def get_fitted_pipeline(self):
        return self.pipeline

    @guide
    def get_pipeline_hyperparameters(self):
        return self.pipeline_hyperparameters

    @guide
    def predict(self, X=None, visual=False, **kwargs):
        if X is None:
            X = self.X_test
        if visual:
            outputs_spec, visual_names = self._get_outputs_spec()
        else:
            outputs_spec = "default"


        outputs = self.pipeline.predict(X, output_=outputs_spec, **kwargs)
        print(outputs)

        if visual and visual_names:
            prediction = outputs[0]
            return prediction, dict(zip(visual_names, outputs[-len(visual_names) :]))

        return outputs

    @guide
    def evaluate(self, X=None, y=None,metrics=None, additional_args = None, context_mapping = None, metric_args_mapping = None):
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        # may have multiple proba_steps and multiple produce args
      
        # context_0 = self.pipeline.predict(X, output_=0)
        # y_proba = context_0["y_pred"][::,  1]
        final_context = self.pipeline.predict(X, output_=-1)

        if metrics is None:
            metrics = DEFAULT_METRICS
        if metric_args is None:
            metric_args = {}

        results = {}
        for metric in metrics:
            try:
                metric_primitive = self._get_ml_primitive(metric)
                additional_kwargs = {}
                if metric_primitive.name in metric_args:
                    additional_kwargs = metric_args[metric_primitive.name]

                res = metric_primitive.produce(y_true = self.y_test, **final_context, **additional_kwargs)
                results[metric_primitive.name] = res
            except Exception as e:
                LOGGER.error(f"Unable to run evaluation metric: {metric_primitive.name}", exc_info = e)
        self.results = results
        return results


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

    # obj.set_entityset(entityset_path = "/Users/raymondpan/zephyr/Zephyr-repo/brake_pad_es", es_type = 'scada')
    
    # obj.set_labeling_function(name="brake_pad_presence")

    obj.generate_label_times(labeling_fn="brake_pad_presence", num_samples=10, gap="20d")
    # print(obj.get_label_times())


    obj.generate_feature_matrix_and_labels(
        target_dataframe_name="turbines",
        cutoff_time_in_index=True,
        agg_primitives=["count", "sum", "max"],
        verbose = True
    )

    print(obj.get_feature_matrix_and_labels)

    obj.generate_train_test_split()
    add_primitives_path(
        path="/Users/raymondpan/zephyr/Zephyr-repo/zephyr_ml/primitives/jsons"
    )
    obj.set_and_fit_pipeline()


    obj.evaluate()
