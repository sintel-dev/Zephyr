#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for zephyr_ml package."""
from datetime import datetime, time

import numpy as np
import pandas as pd

STOPPAGES_DATA = {
    'stoppage_id': [0, 1],
    'WTG': ['A001', 'A001'],
    'Start': [pd.Timestamp('2019-08-01 00:00:00.123000'),
              pd.Timestamp('2019-09-01 00:00:00.456000')],
    'End': [pd.Timestamp('2019-08-23 23:59:59.123000'),
            pd.Timestamp('2019-09-19 01:02:03.456000')],
    'Duration (hh:mi:ss)': [datetime(1900, 1, 30, 3, 4, 5),
                            time(14, 15, 16)],
    'Status': [np.nan, np.nan],
    'Cause': ['Cause', 'Cause'],
    'Origin': [np.nan, np.nan],
    'Class': [np.nan, np.nan],
    'Alarm': [np.nan, np.nan],
    'Est. Lost Gen (kWh)': [1234.0, 45678.0],
    'Comments': ['Yaw Error Stoppage', 'Comment'],
    'SAP W. Order': [np.nan, np.nan],
    'SAP W. Order Name': [np.nan, np.nan],
    'Alarms CORE': [' ', ' ']
}

WORK_ORDERS_DATA = {
    'Order reference timestamp': [pd.Timestamp('2019-07-27 23:59:59'),
                                  pd.Timestamp('2019-08-02 13:21:01')],
    'Order basic start timestamp': [pd.Timestamp('2019-07-27 12:09:41'),
                                    pd.Timestamp('2019-08-02 03:04:56')],
    'Order basic finish timestamp': [pd.Timestamp('2019-08-04 09:08:13'),
                                     pd.Timestamp('2019-08-28 01:28:23')],
    'Order': [123456789.0, 987654321.0],
    'Order created on date': [pd.Timestamp('2019-07-27 00:00:00'),
                              pd.Timestamp('2019-08-02 00:00:00')],
    'Order actual release date': [pd.Timestamp('2019-07-27 00:00:00'),
                                  pd.Timestamp('2019-08-02 00:00:00')],
    'Order actual start date': [pd.Timestamp(np.nan), pd.Timestamp('2019-04-21 00:00:00')],
    'Order actual finish date': [pd.Timestamp(np.nan), pd.Timestamp(np.nan)],
    'Maintenance activity type': ['MPI', 'MPI'],
    'Maintenance activity description': ['Yaw Maintenance', 'Description'],
    'Functional location': ['1234A001123', '2345A001ABC'],
    'Functional location description': ['Func Description 1', 'Func Description 2'],
    'Notification': [1000101115.0, 1000201113.0]
}

NOTIFICATIONS_DATA = {
    'Notification timestamp': [pd.Timestamp('2019-07-28 02:12:34'),
                               pd.Timestamp('2019-08-02 14:12:34')],
    'Malfunction start timestamp': [pd.Timestamp('2019-07-29 00:01:23'),
                                    pd.Timestamp('2019-08-03 00:04:56')],
    'Malfunction end timestamp': [pd.Timestamp('2019-08-03 11:44:23'),
                                  pd.Timestamp('2019-08-20 10:22:16')],
    'Notification created timestamp': [pd.Timestamp('2019-07-30 03:12:34'),
                                       pd.Timestamp('2019-08-04 15:12:34')],
    'Notification changed timestamp': [pd.Timestamp('2019-09-03 05:06:07'),
                                       pd.Timestamp('2019-10-20 05:06:07')],
    'Required start timestamp': [pd.Timestamp('2019-07-28 02:27:06'),
                                 pd.Timestamp('2019-08-02 14:30:44')],
    'Required end timestamp': [pd.Timestamp('2019-07-30 10:19:12'),
                               pd.Timestamp('2019-08-04 11:45:14')],
    'Completion timestamp': [pd.Timestamp('2019-09-03 04:05:06'),
                             pd.Timestamp('2019-10-20 04:05:06')],
    'Order': [123456789.0, 987654321.0],
    'Notification': [1000101115.0, 1000201113.0],
    'Breakdown (X=yes)': ['X', 'nan'],
    'Breakdown duration (hrs)': [12.0, 13.0],
    'Technical object description': ['nan', 'nan'],
    'Material description': ['nan', 'nan'],
    'Assembly description': ['nan', 'nan'],
    'Long text available (X=yes)': ['nan', 'nan'],
    'Subject long text': ['nan', 'nan'],
    'Equipment number': [np.nan, np.nan],
    'Assembly number': [np.nan, np.nan],
    'Priority code': ['P', 'P'],
    'Priority': ['Priority', 'Priority'],
    'Serial number': ['nan', 'nan'],
    'Material number': [np.nan, np.nan],
    'Causing element description': ['nan', 'nan'],
    'Fault mode description': ['nan', 'nan'],
    'Cause of malfunction description': ['nan', 'nan'],
    'Subject description': ['nan', 'Description'],
    'Functional location': ['1234A001123', '2345A001ABC'],
    'Functional location description': ['Func Description 1', 'Func Description 2'],
}


def merge_work_orders_notifications_data():
    """Helper function to merge work orders and notifications data."""
    changed_wo_data = WORK_ORDERS_DATA.copy()
    changed_wo_data['WTG'] = ['A001', 'A001']
    changed_notif_data = NOTIFICATIONS_DATA.copy()
    # matching the output of the merge
    changed_notif_data['Functional location_y'] = changed_notif_data.pop('Functional location')
    changed_notif_data['Functional location description_y'] = (
        changed_notif_data.pop('Functional location description'))
    # matching the notifications update
    changed_wo_data.update(changed_notif_data)
    return changed_wo_data


def merge_label_generation_data():
    expected_data = STOPPAGES_DATA.copy()
    expected_data['stoppage_id'] = [0, 1]
    expected_won = merge_work_orders_notifications_data()

    for key, value in expected_won.items():
        if key not in expected_data:
            expected_data[key] = [expected_won[key][1], np.nan]

    return expected_data
