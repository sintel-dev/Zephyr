# Default EntitySet keyword arguments for entities
DEFAULT_ES_KWARGS = {
    'alarms': {
        'index': '_index',
        'make_index': True,
        'time_index': 'DAT_START',
        'secondary_time_index': {'DAT_END': ['IND_DURATION']},
        'logical_types': {
            'COD_ELEMENT': 'categorical',  # turbine id
            'DAT_START': 'datetime',  # start
            'DAT_END': 'datetime',  # end
            'IND_DURATION': 'double',  # duration
            'COD_ALARM': 'categorical',  # alarm code
            'COD_ALARM_INT': 'categorical',  # international alarm code
            'DES_NAME': 'categorical',  # alarm name
            'DES_TITLE': 'categorical',  # alarm description
            'COD_STATUS': 'categorical'  # status code
        }
    },
    'stoppages': {
        'index': '_index',
        'make_index': True,
        'time_index': 'DAT_START',
        'secondary_time_index': {'DAT_END': ['IND_DURATION', 'IND_LOST_GEN']},
        'logical_types': {
            'COD_ELEMENT': 'categorical',  # turbine id
            'DAT_START': 'datetime',  # start
            'DAT_END': 'datetime',  # end
            'DES_WO_NAME': 'natural_language',  # work order name
            'DES_COMMENTS': 'natural_language',  # work order comments
            'COD_WO': 'integer_nullable',  # stoppage code
            'IND_DURATION': 'double',  # duration
            'IND_LOST_GEN': 'double',  # generation loss
            'COD_ALARM': 'categorical',  # alarm code
            'COD_CAUSE': 'categorical',  # stoppage cause
            'COD_INCIDENCE': 'categorical',  # incidence code
            'COD_ORIGIN': 'categorical',  # origin code
            'DESC_CLASS': 'categorical',  # ????
            'COD_STATUS': 'categorical',  # status code
            'COD_CODE': 'categorical',  # stoppage code
            'DES_DESCRIPTION': 'natural_language',  # stoppage description
            'DES_TECH_NAME': 'categorical'  # turbine technology
        }
    },
    'notifications': {
        'index': '_index',
        'make_index': True,
        'time_index': 'DAT_POSTING',
        'secondary_time_index': {'DAT_MALF_END': ['IND_BREAKDOWN_DUR']},
        'logical_types': {
            'COD_ELEMENT': 'categorical',  # turbine id
            'COD_ORDER': 'categorical',
            'IND_QUANTITY': 'double',
            'COD_MATERIAL_SAP': 'categorical',
            'DAT_POSTING': 'datetime',
            'COD_MAT_DOC': 'categorical',
            'DES_MEDIUM': 'categorical',
            'COD_NOTIF': 'categorical',
            'DAT_MALF_START': 'datetime',
            'DAT_MALF_END': 'datetime',
            'IND_BREAKDOWN_DUR': 'double',
            'FUNCT_LOC_DES': 'categorical',
            'COD_ALARM': 'categorical',
            'DES_ALARM': 'categorical'
        }
    },
    'work_orders': {
        'index': 'COD_ORDER',
        'time_index': 'DAT_BASIC_START',
        'secondary_time_index': {'DAT_VALID_END': []},
        'logical_types': {
            'COD_ELEMENT': 'categorical',
            'COD_ORDER': 'categorical',
            'DAT_BASIC_START': 'datetime',
            'DAT_BASIC_END': 'datetime',
            'COD_EQUIPMENT': 'categorical',
            'COD_MAINT_PLANT': 'categorical',
            'COD_MAINT_ACT_TYPE': 'categorical',
            'COD_CREATED_BY': 'categorical',
            'COD_ORDER_TYPE': 'categorical',
            'DAT_REFERENCE': 'datetime',
            'DAT_CREATED_ON': 'datetime',
            'DAT_VALID_END': 'datetime',
            'DAT_VALID_START': 'datetime',
            'COD_SYSTEM_STAT': 'categorical',
            'DES_LONG': 'natural_language',
            'COD_FUNCT_LOC': 'categorical',
            'COD_NOTIF_OBJ': 'categorical',
            'COD_MAINT_ITEM': 'categorical',
            'DES_MEDIUM': 'natural_language',
            'DES_FUNCT_LOC': 'categorical'
        }
    },
    'turbines': {
        'index': 'COD_ELEMENT',
        'logical_types': {
            'COD_ELEMENT': 'categorical',
            'TURBINE_PI_ID': 'categorical',
            'TURBINE_LOCAL_ID': 'categorical',
            'TURBINE_SAP_COD': 'categorical',
            'DES_CORE_ELEMENT': 'categorical',
            'SITE': 'categorical',
            'DES_CORE_PLANT': 'categorical',
            'COD_PLANT_SAP': 'categorical',
            'PI_COLLECTOR_SITE_NAME': 'categorical',
            'PI_LOCAL_SITE_NAME': 'categorical'
        }
    }
}

DEFAULT_ES_TYPE_KWARGS = {
    'pidata': {
        'index': '_index',
        'make_index': True,
        'time_index': 'time',
        'logical_types': {
            'time': 'datetime',
            'COD_ELEMENT': 'categorical'
        }
    },
    'scada': {
        'index': '_index',
        'make_index': True,
        'time_index': 'TIMESTAMP',
        'logical_types': {
            'TIMESTAMP': 'datetime',
            'COD_ELEMENT': 'categorical'
        }
    }
}


def get_mapped_kwargs(es_type, new_kwargs=None):
    if es_type not in DEFAULT_ES_TYPE_KWARGS.keys():
        raise ValueError('Unrecognized es_type argument: {}'.format(es_type))
    mapped_kwargs = DEFAULT_ES_KWARGS.copy()
    mapped_kwargs.update({es_type: DEFAULT_ES_TYPE_KWARGS[es_type]})

    if new_kwargs is not None:
        if not isinstance(new_kwargs, dict):
            raise ValueError('new_kwargs must be dictionary mapping entity name to dictionary '
                             'with updated keyword arguments for EntitySet creation.')
        for entity in new_kwargs:
            if entity not in mapped_kwargs:
                raise ValueError('Unrecognized entity "{}" found in new keyword argument '
                                 'mapping.'.format(entity))

            mapped_kwargs[entity].update(new_kwargs[entity])

    return mapped_kwargs
