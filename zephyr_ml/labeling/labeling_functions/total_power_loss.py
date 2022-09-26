
from zephyr_ml.labeling.utils import denormalize


def total_power_loss(es, column_map={}):
    """Calculates the total power loss over the data slice.

    Args:
        es (ft.EntitySet):
            EntitySet of data to calculate total power loss across.
        column_map (dict):
            Optional dictionary to update default column names to the
            actual corresponding column names in the data slice. Can contain the
            following keys:
                "lost_gen": Column that contains the generation lost due to stoppage. Defaults
                    to "IND_LOST_GEN".
                "turbine_id": Column containing the ID of the turbine associated with a
                    stoppage. Must match the index column of the 'turbines' entity.
                    Defaults to "COD_ELEMENT".
                "time_index": Column to use as the time index for the data slice.
                    Defaults to "DAT_START".

    Returns:
        label:
            Labeling function to find the total power loss over a data slice.
        df:
            Denormalized dataframe of data to get labels from
        meta:
            Dictionary containing metadata about labeling function

    """
    lost_gen = column_map.get('lost_gen', 'IND_LOST_GEN')
    turbine_id = column_map.get('turbine_id_column', 'COD_ELEMENT')
    time_index = column_map.get('time_index', 'DAT_START')

    def label(ds, **kwargs):
        return sum(ds[lost_gen])

    meta = {
        "target_entity_index": turbine_id,
        "time_index": time_index,
    }

    df = denormalize(es, entities=['stoppages'])

    return label, df, meta
