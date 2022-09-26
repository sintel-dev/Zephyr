import pandas as pd
import pytest

from zephyr_ml import create_pidata_entityset, create_scada_entityset


@pytest.fixture
def base_dfs():
    alarms_df = pd.DataFrame({
        'COD_ELEMENT': [0, 0],
        'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 11:12:13')],
        'DAT_END': [pd.Timestamp('2022-01-01 13:00:00'), pd.Timestamp('2022-03-02 11:12:13')],
        'IND_DURATION': [0.5417, 1.0],
        'COD_ALARM': [12345, 98754],
        'COD_ALARM_INT': [12345, 98754],
        'DES_NAME': ['Alarm1', 'Alarm2'],
        'DES_TITLE': ['Description of alarm 1', 'Description of alarm 2'],
    })
    stoppages_df = pd.DataFrame({
        'COD_ELEMENT': [0, 0],
        'DAT_START': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 11:12:13')],
        'DAT_END': [pd.Timestamp('2022-01-08 11:07:17'), pd.Timestamp('2022-03-01 17:00:13')],
        'DES_WO_NAME': ['stoppage name 1', 'stoppage name 2'],
        'DES_COMMENTS': ['description of stoppage 1', 'description of stoppage 2'],
        'COD_WO': [12345, 67890],
        'IND_DURATION': [7.4642, 0.2417],
        'IND_LOST_GEN': [45678.0, 123.0],
        'COD_ALARM': [12345, 12345],
        'COD_CAUSE': [32, 48],
        'COD_INCIDENCE': [987654, 123450],
        'COD_ORIGIN': [6, 23],
        'COD_STATUS': ['STOP', 'PAUSE'],
        'COD_CODE': ['ABC', 'XYZ'],
        'DES_DESCRIPTION': ['Description 1', 'Description 2']
    })
    notifications_df = pd.DataFrame({
        'COD_ELEMENT': [0, 0],
        'COD_ORDER': [12345, 67890],
        'IND_QUANTITY': [1, -20],
        'COD_MATERIAL_SAP': [36052411, 67890],
        'DAT_POSTING': [pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-03-01 00:00:00')],
        'COD_MAT_DOC': [77889900, 12345690],
        'DES_MEDIUM': ['Description of notification 1', 'Description of notification 2'],
        'COD_NOTIF': [567890123, 32109877],
        'DAT_MALF_START': [pd.Timestamp('2021-12-25 18:07:10'),
                           pd.Timestamp('2022-02-28 06:04:00')],
        'DAT_MALF_END': [pd.Timestamp('2022-01-08 11:07:17'), pd.Timestamp('2022-03-01 17:00:13')],
        'IND_BREAKDOWN_DUR': [14.1378, 2.4792],
        'FUNCT_LOC_DES': ['location description 1', 'location description 2'],
        'COD_ALARM': [12345, 12345],
        'DES_ALARM': ['Alarm description', 'Alarm description'],
    })
    work_orders_df = pd.DataFrame({
        'COD_ELEMENT': [0, 0],
        'COD_ORDER': [12345, 67890],
        'DAT_BASIC_START': [pd.Timestamp('2022-01-01 00:00:00'),
                            pd.Timestamp('2022-03-01 00:00:00')],
        'DAT_BASIC_END': [pd.Timestamp('2022-01-09 00:00:00'),
                          pd.Timestamp('2022-03-02 00:00:00')],
        'COD_EQUIPMENT': [98765, 98765],
        'COD_MAINT_PLANT': ['ABC', 'ABC'],
        'COD_MAINT_ACT_TYPE': ['XYZ', 'XYZ'],
        'COD_CREATED_BY': ['A1234', 'B6789'],
        'COD_ORDER_TYPE': ['A', 'B'],
        'DAT_REFERENCE': [pd.Timestamp('2022-01-01 00:00:00'),
                          pd.Timestamp('2022-03-01 00:00:00')],
        'DAT_CREATED_ON': [pd.Timestamp('2022-03-01 00:00:00'),
                           pd.Timestamp('2022-04-18 00:00:00')],
        'DAT_VALID_END': [pd.NaT, pd.NaT],
        'DAT_VALID_START': [pd.NaT, pd.NaT],
        'COD_SYSTEM_STAT': ['ABC XYZ', 'LMN OPQ'],
        'DES_LONG': ['description of work order', 'description of work order'],
        'COD_FUNCT_LOC': ['!12345', '?09876'],
        'COD_NOTIF_OBJ': ['00112233', '00998877'],
        'COD_MAINT_ITEM': ['', '019283'],
        'DES_MEDIUM': ['short description', 'short description'],
        'DES_FUNCT_LOC': ['XYZ1234', 'ABC9876'],
    })
    turbines_df = pd.DataFrame({
        'COD_ELEMENT': [0],
        'TURBINE_PI_ID': ['TA00'],
        'TURBINE_LOCAL_ID': ['A0'],
        'TURBINE_SAP_COD': ['LOC000'],
        'DES_CORE_ELEMENT': ['T00'],
        'SITE': ['LOCATION'],
        'DES_CORE_PLANT': ['LOC'],
        'COD_PLANT_SAP': ['ABC'],
        'PI_COLLECTOR_SITE_NAME': ['LOC0'],
        'PI_LOCAL_SITE_NAME': ['LOC0']
    })
    return {
        'alarms': alarms_df,
        'stoppages': stoppages_df,
        'notifications': notifications_df,
        'work_orders': work_orders_df,
        'turbines': turbines_df
    }


@pytest.fixture
def pidata_dfs(base_dfs):
    pidata_df = pd.DataFrame({
        'time': [pd.Timestamp('2022-01-02 13:21:01'), pd.Timestamp('2022-03-08 13:21:01')],
        'COD_ELEMENT': [0, 0],
        'val1': [9872.0, 559.0],
        'val2': [10.0, -7.0]
    })
    return {**base_dfs, 'pidata': pidata_df}


@pytest.fixture
def scada_dfs(base_dfs):
    scada_df = pd.DataFrame({
        'TIMESTAMP': [pd.Timestamp('2022-01-02 13:21:01'), pd.Timestamp('2022-03-08 13:21:01')],
        'COD_ELEMENT': [0, 0],
        'val1': [1002.0, 56.8],
        'val2': [-98.7, 1004.2]
    })
    return {**base_dfs, 'scada': scada_df}


def test_create_pidata_missing_entities(pidata_dfs):
    error_msg = 'Missing dataframes for entities notifications.'

    pidata_dfs.pop('notifications')
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_create_scada_missing_entities(scada_dfs):
    error_msg = 'Missing dataframes for entities notifications.'

    scada_dfs.pop('notifications')
    with pytest.raises(ValueError, match=error_msg):
        create_scada_entityset(scada_dfs)


def test_create_pidata_extra_entities(pidata_dfs):
    error_msg = "Unrecognized entities extra included in dfs."

    pidata_dfs['extra'] = pd.DataFrame()
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_create_scada_extra_entities(scada_dfs):
    error_msg = "Unrecognized entities extra included in dfs."

    scada_dfs['extra'] = pd.DataFrame()
    with pytest.raises(ValueError, match=error_msg):
        create_scada_entityset(scada_dfs)


def test_missing_wo_index_columns(pidata_dfs):
    error_msg = 'Expected column "COD_ORDER" missing from notifications entity'
    pidata_dfs['notifications'].drop(columns=['COD_ORDER'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)

    error_msg = 'Expected index column "COD_ORDER" missing from work_orders entity'
    pidata_dfs['work_orders'].drop(columns=['COD_ORDER'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_wo_index_column_nonunique(pidata_dfs):
    error_msg = 'Expected index column "COD_ORDER" of work_orders entity is not unique'

    pidata_dfs['work_orders']['COD_ORDER'] = [12345, 12345]
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_missing_turbine_index_columns(pidata_dfs):
    error_msg = 'Turbines index column "COD_ELEMENT" missing from data for alarms entity'

    pidata_dfs['alarms'].drop(columns='COD_ELEMENT', inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)

    error_msg = 'Expected index column "COD_ELEMENT" missing from turbines entity'

    pidata_dfs['turbines'].drop(columns=['COD_ELEMENT'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_missing_time_indices(pidata_dfs):
    error_msg = 'Column "IND_LOST_GEN" associated with secondary time index ' + \
                '"DAT_END" missing from stoppages entity'
    pidata_dfs['stoppages'].drop(columns=['IND_LOST_GEN'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)

    error_msg = 'Secondary time index "DAT_END" missing from stoppages entity'
    pidata_dfs['stoppages'].drop(columns=['DAT_END'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)

    error_msg = 'Missing time index column "DAT_START" from stoppages entity'
    pidata_dfs['stoppages'].drop(columns=['DAT_START'], inplace=True)
    with pytest.raises(ValueError, match=error_msg):
        create_pidata_entityset(pidata_dfs)


def test_default_create_pidata_entityset(pidata_dfs):
    es = create_pidata_entityset(pidata_dfs)

    assert es.id == 'PI data'
    assert set(es.dataframe_dict.keys()) == set(
        ['alarms', 'turbines', 'stoppages', 'work_orders', 'notifications', 'pidata'])


def test_default_create_scada_entityset(scada_dfs):
    es = create_scada_entityset(scada_dfs)

    assert es.id == 'SCADA data'
    assert set(es.dataframe_dict.keys()) == set(
        ['alarms', 'turbines', 'stoppages', 'work_orders', 'notifications', 'scada'])
