import pandas as pd
import pytest

from zephyr_ml import create_pidata_entityset, create_scada_entityset
from zephyr_ml.feature_engineering import process_signals


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


@pytest.fixture
def pidata_es(pidata_dfs):
    return create_pidata_entityset(pidata_dfs)


@pytest.fixture
def scada_es(scada_dfs):
    return create_scada_entityset(scada_dfs)


@pytest.fixture
def transformations():
    return [{
        "name": "fft",
        "primitive": "sigpro.transformations.amplitude.identity.identity"
    }]


@pytest.fixture
def aggregations():
    return [{
        "name": "mean",
        "primitive": "sigpro.aggregations.amplitude.statistical.mean"
    }]


def test_process_signals_pidata(pidata_es, transformations, aggregations):
    signal_dataframe_name = 'pidata'
    signal_column = 'val1'
    window_size = '1m'
    replace_dataframe = False
    before = pidata_es['pidata'].copy()

    process_signals(pidata_es, signal_dataframe_name, signal_column, transformations, aggregations,
                    window_size, replace_dataframe)

    processed = pidata_es['pidata_processed'].copy()
    after = pidata_es['pidata'].copy()

    expected = pd.DataFrame({
        "_index": [0, 1, 2],
        "COD_ELEMENT": [0, 0, 0],
        "time": [
            pd.Timestamp('2022-01-31'),
            pd.Timestamp('2022-02-28'),
            pd.Timestamp('2022-03-31')
        ],
        "fft.mean.mean_value": [9872, None, 559]
    })
    expected['COD_ELEMENT'] = expected['COD_ELEMENT'].astype('category')
    expected['fft.mean.mean_value'] = expected['fft.mean.mean_value'].astype('float64')
    processed['fft.mean.mean_value'] = processed['fft.mean.mean_value'].astype('float64')

    assert pidata_es['pidata_processed'].shape[0] == 3
    assert pidata_es['pidata_processed'].shape[1] == 4

    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(processed, expected)


def test_process_signals_pidata_replace(pidata_es, transformations, aggregations):
    signal_dataframe_name = 'pidata'
    signal_column = 'val1'
    window_size = '1m'
    replace_dataframe = True

    process_signals(pidata_es, signal_dataframe_name, signal_column, transformations, aggregations,
                    window_size, replace_dataframe)

    processed = pidata_es['pidata'].copy()

    expected = pd.DataFrame({
        "_index": [0, 1, 2],
        "COD_ELEMENT": [0, 0, 0],
        "time": [
            pd.Timestamp('2022-01-31'),
            pd.Timestamp('2022-02-28'),
            pd.Timestamp('2022-03-31')
        ],
        "fft.mean.mean_value": [9872, None, 559]
    })
    expected['COD_ELEMENT'] = expected['COD_ELEMENT'].astype('category')
    expected['fft.mean.mean_value'] = expected['fft.mean.mean_value'].astype('float64')
    processed['fft.mean.mean_value'] = processed['fft.mean.mean_value'].astype('float64')

    assert pidata_es['pidata'].shape[0] == 3
    assert pidata_es['pidata'].shape[1] == 4

    pd.testing.assert_frame_equal(processed, expected)
    assert 'pidata_processed' not in list(pidata_es.dataframe_dict.keys())


def test_process_signals_scada(scada_es, transformations, aggregations):
    signal_dataframe_name = 'scada'
    signal_column = 'val1'
    window_size = '1m'
    replace_dataframe = False
    before = scada_es['scada'].copy()

    process_signals(scada_es, signal_dataframe_name, signal_column, transformations, aggregations,
                    window_size, replace_dataframe)

    expected = pd.DataFrame({
        "_index": [0, 1, 2],
        "COD_ELEMENT": [0, 0, 0],
        "TIMESTAMP": [
            pd.Timestamp('2022-01-31'),
            pd.Timestamp('2022-02-28'),
            pd.Timestamp('2022-03-31')
        ],
        "fft.mean.mean_value": [1002, None, 56.8]
    })
    expected['COD_ELEMENT'] = expected['COD_ELEMENT'].astype('category')
    expected['fft.mean.mean_value'] = expected['fft.mean.mean_value'].astype('float64')
    after = scada_es['scada'].copy()

    assert scada_es['scada_processed'].shape[0] == 3
    assert scada_es['scada_processed'].shape[1] == 4

    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(scada_es['scada_processed'], expected)


def test_process_signals_scada_replace(scada_es, transformations, aggregations):
    signal_dataframe_name = 'scada'
    signal_column = 'val1'
    window_size = '1m'
    replace_dataframe = True

    process_signals(scada_es, signal_dataframe_name, signal_column, transformations, aggregations,
                    window_size, replace_dataframe)

    expected = pd.DataFrame({
        "_index": [0, 1, 2],
        "COD_ELEMENT": [0, 0, 0],
        "TIMESTAMP": [
            pd.Timestamp('2022-01-31'),
            pd.Timestamp('2022-02-28'),
            pd.Timestamp('2022-03-31')
        ],
        "fft.mean.mean_value": [1002, None, 56.8]
    })
    expected['COD_ELEMENT'] = expected['COD_ELEMENT'].astype('category')
    expected['fft.mean.mean_value'] = expected['fft.mean.mean_value'].astype('float64')

    assert scada_es['scada'].shape[0] == 3
    assert scada_es['scada'].shape[1] == 4

    pd.testing.assert_frame_equal(scada_es['scada'], expected)
    assert 'scada_processed' not in list(scada_es.dataframe_dict.keys())
