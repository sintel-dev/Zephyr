from itertools import chain

import featuretools as ft

from zephyr_ml.metadata import get_mapped_kwargs


def _validate_data(dfs, es_type, es_kwargs):
    """Validate data by checking for required columns in each entity"""
    if not isinstance(es_type, list):
        es_type = [es_type]

    entities = set(
        chain(
            [
                "alarms",
                "stoppages",
                "work_orders",
                "notifications",
                "turbines",
                *es_type,
            ]
        )
    )

    if set(dfs.keys()) != entities:
        missing = entities.difference(set(dfs.keys()))
        extra = set(dfs.keys()).difference(entities)
        msg = []
        if missing:
            msg.append("Missing dataframes for entities {}.".format(
                ", ".join(missing)))
        if extra:
            msg.append(
                "Unrecognized entities {} included in dfs.".format(
                    ", ".join(extra))
            )

        raise ValueError(" ".join(msg))

    turbines_index = es_kwargs["turbines"]["index"]
    work_orders_index = es_kwargs["work_orders"]["index"]

    if work_orders_index not in dfs["work_orders"].columns:
        raise ValueError(
            'Expected index column "{}" missing from work_orders entity'.format(
                work_orders_index
            )
        )

    if work_orders_index not in dfs["notifications"].columns:
        raise ValueError(
            'Expected column "{}" missing from notifications entity'.format(
                work_orders_index
            )
        )

    if not dfs["work_orders"][work_orders_index].is_unique:
        raise ValueError(
            'Expected index column "{}" of work_orders entity is not '
            "unique".format(work_orders_index)
        )

    if turbines_index not in dfs["turbines"].columns:
        raise ValueError(
            'Expected index column "{}" missing from turbines entity'.format(
                turbines_index
            )
        )

    if not dfs["turbines"][turbines_index].is_unique:
        raise ValueError(
            'Expected index column "{}" of turbines entity is not unique.'.format(
                turbines_index
            )
        )

    for entity, df in dfs.items():
        if turbines_index not in df.columns:
            raise ValueError(
                'Turbines index column "{}" missing from data for {} entity'.format(
                    turbines_index, entity
                )
            )

        time_index = es_kwargs[entity].get("time_index", False)
        if time_index and time_index not in df.columns:
            raise ValueError(
                'Missing time index column "{}" from {} entity'.format(
                    time_index, entity
                )
            )

        secondary_time_indices = es_kwargs[entity].get(
            "secondary_time_index", {})
        for time_index, cols in secondary_time_indices.items():
            if time_index not in df.columns:
                raise ValueError(
                    'Secondary time index "{}" missing from {} entity'.format(
                        time_index, entity
                    )
                )
            for col in cols:
                if col not in df.columns:
                    raise ValueError(
                        (
                            'Column "{}" associated with secondary time index "{}" '
                            "missing from {} entity"
                        ).format(col, time_index, entity)
                    )


def validate_scada_data(dfs, new_kwargs_mapping=None):
    """
    SCADA data is signal data from the Original Equipment Manufacturer Supervisory Control
    And Data Acquisition (OEM-SCADA) system, a signal data source.
    """
    entity_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
    _validate_data(dfs, "scada", entity_kwargs)
    return entity_kwargs


def validate_pidata_data(dfs, new_kwargs_mapping=None):
    """
    PI data is signal data from the operator's historical Plant Information (PI) system.
    """
    entity_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
    _validate_data(dfs, "pidata", entity_kwargs)
    return entity_kwargs


def validate_vibrations_data(dfs, new_kwargs_mapping=None):
    """
    Vibrations data is vibrations data collected on Planetary gearboxes in turbines.
    """
    entities = ["vibrations"]

    pidata_kwargs, scada_kwargs = {}, {}
    if "pidata" in dfs:
        pidata_kwargs = get_mapped_kwargs("pidata", new_kwargs_mapping)
        entities.append("pidata")
    if "scada" in dfs:
        scada_kwargs = get_mapped_kwargs("scada", new_kwargs_mapping)
        entities.append("scada")

    entity_kwargs = {
        **pidata_kwargs,
        **scada_kwargs,
        **get_mapped_kwargs("vibrations", new_kwargs_mapping),
    }
    _validate_data(dfs, entities, entity_kwargs)
    return entity_kwargs


VALIDATE_DATA_FUNCTIONS = {
    "scada": validate_scada_data,
    "pidata": validate_pidata_data,
    "vibrations": validate_vibrations_data,
}


def _create_entityset(entities, es_type, new_kwargs_mapping=None):

    validate_func = VALIDATE_DATA_FUNCTIONS[es_type]
    es_kwargs = validate_func(entities, new_kwargs_mapping)

    # filter out stated logical types for missing columns
    for entity, df in entities.items():
        es_kwargs[entity]["logical_types"] = {
            col: t
            for col, t in es_kwargs[entity]["logical_types"].items()
            if col in df.columns
        }

    turbines_index = es_kwargs["turbines"]["index"]
    work_orders_index = es_kwargs["work_orders"]["index"]

    relationships = [
        ("turbines", turbines_index, "alarms", turbines_index),
        ("turbines", turbines_index, "stoppages", turbines_index),
        ("turbines", turbines_index, "work_orders", turbines_index),
        ("turbines", turbines_index, es_type, turbines_index),
        ("work_orders", work_orders_index, "notifications", work_orders_index),
    ]

    es = ft.EntitySet()
    es.id = es_type

    for name, df in entities.items():
        es.add_dataframe(dataframe_name=name, dataframe=df, **es_kwargs[name])

    for relationship in relationships:
        parent_df, parent_column, child_df, child_column = relationship
        es.add_relationship(parent_df, parent_column, child_df, child_column)

    return es
