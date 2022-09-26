import featuretools as ft

from zephyr_ml.metadata import get_mapped_kwargs


def _create_entityset(entities, es_type, es_kwargs):
    # filter out stated logical types for missing columns
    for entity, df in entities.items():
        es_kwargs[entity]['logical_types'] = {
            col: t for col, t in es_kwargs[entity]['logical_types'].items()
            if col in df.columns
        }

    turbines_index = es_kwargs['turbines']['index']
    work_orders_index = es_kwargs['work_orders']['index']

    relationships = [
        ('turbines', turbines_index, 'alarms', turbines_index),
        ('turbines', turbines_index, 'stoppages', turbines_index),
        ('turbines', turbines_index, 'work_orders', turbines_index),
        ('turbines', turbines_index, es_type, turbines_index),
        ('work_orders', work_orders_index, 'notifications', work_orders_index)
    ]

    es = ft.EntitySet()

    for name, df in entities.items():
        es.add_dataframe(
            dataframe_name=name,
            dataframe=df,
            **es_kwargs[name]
        )

    for relationship in relationships:
        parent_df, parent_column, child_df, child_column = relationship
        es.add_relationship(parent_df, parent_column, child_df, child_column)

    return es


def create_pidata_entityset(dfs, new_kwargs_mapping=None):
    '''Generate an entityset for PI data datasets

    Args:
        data_paths (dict): Dictionary mapping entity names ('alarms', 'notifications',
            'stoppages', 'work_orders', 'pidata', 'turbines') to the pandas dataframe for
            that entity.
        **kwargs: Updated keyword arguments to be used during entityset creation
    '''
    entity_kwargs = get_mapped_kwargs('pidata', new_kwargs_mapping)
    _validate_data(dfs, 'pidata', entity_kwargs)

    es = _create_entityset(dfs, 'pidata', entity_kwargs)
    es.id = 'PI data'

    return es


def create_scada_entityset(dfs, new_kwargs_mapping=None):
    '''Generate an entityset for SCADA data datasets

    Args:
        data_paths (dict): Dictionary mapping entity names ('alarms', 'notifications',
            'stoppages', 'work_orders', 'scada', 'turbines') to the pandas dataframe for
            that entity.
    '''
    entity_kwargs = get_mapped_kwargs('scada', new_kwargs_mapping)
    _validate_data(dfs, 'scada', entity_kwargs)

    es = _create_entityset(dfs, 'scada', entity_kwargs)
    es.id = 'SCADA data'

    return es


def _validate_data(dfs, es_type, es_kwargs):
    '''Validate data by checking for required columns in each entity
    '''
    entities = set(['alarms', 'stoppages', 'work_orders', 'notifications', 'turbines', es_type])
    if set(dfs.keys()) != entities:
        missing = entities.difference(set(dfs.keys()))
        extra = set(dfs.keys()).difference(entities)
        msg = []
        if missing:
            msg.append('Missing dataframes for entities {}.'.format(', '.join(missing)))
        if extra:
            msg.append('Unrecognized entities {} included in dfs.'.format(', '.join(extra)))

        raise ValueError(' '.join(msg))

    turbines_index = es_kwargs['turbines']['index']
    work_orders_index = es_kwargs['work_orders']['index']

    if work_orders_index not in dfs['work_orders'].columns:
        raise ValueError(
            'Expected index column "{}" missing from work_orders entity'.format(work_orders_index))

    if work_orders_index not in dfs['notifications'].columns:
        raise ValueError(
            'Expected column "{}" missing from notifications entity'.format(work_orders_index))

    if not dfs['work_orders'][work_orders_index].is_unique:
        raise ValueError('Expected index column "{}" of work_orders entity is not '
                         'unique'.format(work_orders_index))

    if turbines_index not in dfs['turbines'].columns:
        raise ValueError(
            'Expected index column "{}" missing from turbines entity'.format(turbines_index))

    if not dfs['turbines'][turbines_index].is_unique:
        raise ValueError(
            'Expected index column "{}" of turbines entity is not unique.'.format(turbines_index))

    for entity, df in dfs.items():
        if turbines_index not in df.columns:
            raise ValueError(
                'Turbines index column "{}" missing from data for {} entity'.format(
                    turbines_index, entity))

        time_index = es_kwargs[entity].get('time_index', False)
        if time_index and time_index not in df.columns:
            raise ValueError(
                'Missing time index column "{}" from {} entity'.format(
                    time_index, entity))

        secondary_time_indices = es_kwargs[entity].get('secondary_time_index', {})
        for time_index, cols in secondary_time_indices.items():
            if time_index not in df.columns:
                raise ValueError(
                    'Secondary time index "{}" missing from {} entity'.format(
                        time_index, entity))
            for col in cols:
                if col not in df.columns:
                    raise ValueError(('Column "{}" associated with secondary time index "{}" '
                                     'missing from {} entity').format(col, time_index, entity))
