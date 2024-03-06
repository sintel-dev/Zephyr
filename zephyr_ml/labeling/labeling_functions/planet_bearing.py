from zephyr_ml.labeling.utils import denormalize


def gearbox_replace_presence(es, column_map={}):
    """Determines if gearbox replacement/exchange is present in stoppages.

    Args:
        es (ft.EntitySet):
            EntitySet of data to check gearbox replacements.
        column_map (dict):
            Optional dictionary to update default column names to the
            actual corresponding column names in the data slice. Can contain the
            following keys:
                "comments": Column that contains comments about the stoppage. Defaults
                    to "DES_COMMENTS".
                "turbine_id": Column containing the ID of the turbine associated with a
                    stoppage. Must match the index column of the 'turbines' entity.
                    Defaults to "COD_ELEMENT".
                "time_index": Column to use as the time index for the data slice.
                    Defaults to "DAT_END".

    Returns:
        label:
            Labeling function to find gearbox replacement presence over a data slice.
        df:
            Denormalized dataframe of data to get labels from.
        meta:
            Dictionary containing metadata about labeling function.

    """
    comments = column_map.get('comments_column', 'DES_COMMENTS')
    turbine_id = column_map.get('turbine_id_column', 'COD_ELEMENT')
    time_index = column_map.get('time_index_column', 'DAT_END')

    def label(ds, **kwargs):
        label_strings = ['Gearbox replace*', 'Gearbox exchange']
        comments_lower = ds[comments].fillna('').str.lower()
        f = any(comments_lower.str.contains('|'.join(label_strings), case=False))
        return f

    meta = {
        "target_entity_index": turbine_id,
        "time_index": time_index,
    }

    df = denormalize(es, entities=['stoppages'])

    return label, df, meta
