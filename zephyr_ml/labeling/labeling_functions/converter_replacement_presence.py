from zephyr_ml.labeling.utils import denormalize


def converter_replacement_presence(es, column_map={}):
    """Calculates the converter replacement presence.

    Args:
        es (ft.EntitySet):
            EntitySet of data to check converter replacements.
        column_map (dict):
            Optional dictionary to update default column names to the
            actual corresponding column names in the data slice. Can contain the
            following keys:
                "sap_code": Column that contains the material SAP code. Defaults
                    to "COD_MATERIAL_SAP".
                "turbine_id": Column containing the ID of the turbine associated with a
                    stoppage. Must match the index column of the 'turbines' entity.
                    Defaults to "COD_ELEMENT".
                "description": Column containing the description for a given notification.
                    Defaults to "DES_MEDIUM".
                "time_index": Column to use as the time index for the data slice.
                    Defaults to "DAT_MALF_START".


    Returns:
        label:
            Labeling function to find converter replacement presence over a data slice.
        df:
            Denormalized dataframe of data to get labels from.
        meta:
            Dictionary containing metadata about labeling function.

    """
    sap_code = column_map.get('sap_code', 'COD_MATERIAL_SAP')
    column_map.get('description', 'DES_MEDIUM')
    turbine_id = column_map.get('turbine_id_column', 'COD_ELEMENT')
    time_index = column_map.get('time_index', 'DAT_MALF_START')

    def label(ds, **kwargs):
        logic1 = (ds[sap_code] == 36052411).any()
        # logic2 = ds[DESCRIPTION].str.lower().apply(lambda x: 'inu' in x).any()
        f = logic1  # or logic2
        return f

    meta = {
        "target_entity_index": turbine_id,
        "time_index": time_index,
        "window_size": "10d"
    }

    # denormalize(es, entities=['notifications', 'work_orders'])
    df = denormalize(es, entities=['notifications'])
    df = df.dropna(subset=[time_index])

    return label, df, meta
