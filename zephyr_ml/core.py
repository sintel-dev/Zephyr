from zephyr_ml.metadata import get_default_es_type_kwargs
from zephyr_ml.entityset import get_create_entityset_functions


class Zephyr:

    def __init__(self, pipeline, hyperparameters):
        self.pipeline = pipeline
        self.hyperparameters = hyperparameters

    def get_entityset_types(self):
        """
        Returns the supported entityset types (PI/SCADA) and the required dataframes and their columns
        """
        return get_default_es_type_kwargs()

    def create_entityset(self, data_paths, es_type="scada", new_kwargs_mapping=None):
        """
        Generate an entityset

        Args:
        data_paths ( dict ): Dictionary mapping entity names to the pandas
        dataframe for that that entity
        es_type (str): type of signal data , either SCADA or PI
        new_kwargs_mapping ( dict ): Updated keyword arguments to be used
        during entityset creation
        Returns :
        featuretools . EntitySet that contains the data passed in and
        their relationships
        """
        create_entityset_functions = get_create_entityset_functions()
        if es_type not in create_entityset_functions:
            raise ValueError("Unrecognized es_type argument: {}".format(es_type))

        _create_entityset = create_entityset_functions[es_type]
        entityset = _create_entityset(data_paths, new_kwargs_mapping)
        self.entityset = entityset
        return self.entityset

    def get_entityset(self):
        if self.entityset is None:
            raise

        return self.entityset
