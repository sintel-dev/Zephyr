import pytest

from zephyr_ml.metadata import DEFAULT_ES_KWARGS, DEFAULT_ES_TYPE_KWARGS, get_mapped_kwargs


def test_default_scada_mapped_kwargs():
    expected = {**DEFAULT_ES_KWARGS, 'scada': DEFAULT_ES_TYPE_KWARGS['scada']}
    actual = get_mapped_kwargs('scada')
    assert expected == actual


def test_default_pidata_mapped_kwargs():
    expected = {**DEFAULT_ES_KWARGS, 'pidata': DEFAULT_ES_TYPE_KWARGS['pidata']}
    actual = get_mapped_kwargs('pidata')
    assert expected == actual


def test_new_kwargs_bad_es_type():
    error_text = "Unrecognized es_type argument: bad_es_type"
    with pytest.raises(ValueError, match=error_text):
        get_mapped_kwargs('bad_es_type')


def test_new_kwargs_bad_format():
    error_text = "new_kwargs must be dictionary mapping entity name to dictionary " + \
                 "with updated keyword arguments for EntitySet creation."
    bad_kwargs = ['list', 'of', 'args']
    with pytest.raises(ValueError, match=error_text):
        get_mapped_kwargs('pidata', bad_kwargs)


def test_new_kwargs_unexpected_entity():
    error_text = 'Unrecognized entity "unexpected" found in new keyword argument mapping.'
    bad_kwargs = {'unexpected': {}}
    with pytest.raises(ValueError, match=error_text):
        get_mapped_kwargs('pidata', bad_kwargs)


def test_new_kwargs_update_kwargs():
    updated_kwargs = {'turbines': {
        'index': 'new_turbine_index',
        'logical_types': {
            'COD_ELEMENT': 'integer',
            'extra_column': 'double'
        }
    }}

    expected = {
        **DEFAULT_ES_KWARGS,
        'pidata': DEFAULT_ES_TYPE_KWARGS['pidata'],
        'turbines': {
            'index': 'new_turbine_index',
            'logical_types': {
                'COD_ELEMENT': 'integer',
                'extra_column': 'double'
            }
        }
    }

    actual = get_mapped_kwargs('pidata', updated_kwargs)
    assert actual == expected
