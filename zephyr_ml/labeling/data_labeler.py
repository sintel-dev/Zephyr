from inspect import getfullargspec

import composeml as cp


class DataLabeler:
    """Class that defines the prediction problem.
    This class supports the generation of `label_times` which
    is fundamental to the feature generation phase as well
    as specifying the target labels.
    Args:
        function (LabelingFunction):
            function that defines the labeling function, it should return a
            tuple of labeling function, the dataframe, and the name of the
            target entity.
    """

    def __init__(self, function):
        self.function = function

    def generate_label_times(self, es, num_samples=-1, subset=None,
                             column_map={}, verbose=False, **kwargs):
        """Searches the data to calculate label times.
          Args:
              es (featuretools.EntitySet):
                  Entityset to extract `label_times` from.
              num_samples (int):
                  Number of samples for each to return. Defaults to -1 which returns all possible
                  samples.
              subset (float or int):
                  Portion of the data to select for searching.
              verbose:
                  An indicator to the verbosity of searching.
              column_map:
                  Dictionary mapping column references in labeling function to actual column names.
                  See labeling function for columns referenced.
          Returns:
              composeml.LabelTimes:
                  Calculated labels with cutoff times.
        """
        labeling_function, df, meta = self.function(es, column_map)

        data = df
        if isinstance(subset, float) or isinstance(subset, int):
            data = data.sample(subset)

        target_entity_index = meta.get('target_entity_index')
        time_index = meta.get('time_index')
        thresh = kwargs.get('thresh') or meta.get('thresh')
        window_size = kwargs.get('window_size') or meta.get('window_size')
        label_maker = cp.LabelMaker(labeling_function=labeling_function,
                                    target_dataframe_name=target_entity_index,
                                    time_index=time_index,
                                    window_size=window_size)

        kwargs = {**meta, **kwargs}
        kwargs = {
            k: kwargs.get(k) for k in set(
                getfullargspec(
                    label_maker.search)[0]) if kwargs.get(k) is not None}
        label_times = label_maker.search(data.sort_values(time_index), num_samples,
                                         verbose=verbose, **kwargs)
        if thresh is not None:
            label_times = label_times.threshold(thresh)

        return label_times, meta
