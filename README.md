<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
<i>A project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/zephyr_ml.svg)](https://pypi.python.org/pypi/zephyr_ml)-->
<!--[![Downloads](https://pepy.tech/badge/zephyr_ml)](https://pepy.tech/project/zephyr_ml)-->
<!--[![Travis CI Shield](https://travis-ci.org/signals-dev/zephyr.svg?branch=main)](https://travis-ci.org/signals-dev/zephyr)-->
<!--[![Coverage Status](https://codecov.io/gh/signals-dev/zephyr/branch/main/graph/badge.svg)](https://codecov.io/gh/signals-dev/zephyr)-->

# Zephyr

A machine learning library for assisting in the generation of machine learning problems for wind farms operations data by analyzing past occurrences of events.

 | Important Links                     |                                                                      |
 | ----------------------------------- | -------------------------------------------------------------------- |
 | :computer: **[Website]**            | Check out the Sintel Website for more information about the project. |
 | :book: **[Documentation]**          | Quickstarts, User and Development Guides, and API Reference.         |
 | :star: **[Tutorials]**              | Checkout our notebooks                                               |
 | :octocat: **[Repository]**          | The link to the Github Repository of this library.                   |
 | :scroll: **[License]**              | The repository is published under the MIT License.                   |
 | :keyboard: **[Development Status]** | This software is in its Pre-Alpha stage.                             |
 | ![][Slack Logo] **[Community]**    | Join our Slack Workspace for announcements and discussions.          |

 [Website]: https://sintel.dev/
 [Documentation]: https://dtail.gitbook.io/zephyr/
 [Repository]: https://github.com/sintel-dev/Zephyr
 [License]: https://github.com/sintel-dev/Zephyr/blob/master/LICENSE
 [Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
 [Community]: https://join.slack.com/t/sintel-space/shared_invite/zt-q147oimb-4HcphcxPfDAM0O9_4PaUtw
 [Slack Logo]: https://github.com/sintel-dev/Orion/blob/master/docs/images/slack.png

 - Homepage: https://github.com/signals-dev/zephyr

# Overview

The **Zephyr** library is a framework designed to assist in the
generation of machine learning problems for wind farms operations data by analyzing past
occurrences of events.

The main features of **Zephyr** are:

* **EntitySet creation**: tools designed to represent wind farm data and the relationship
between different tables. We have functions to create EntitySets for datasets with PI data
and datasets using SCADA data.
* **Labeling Functions**: a collection of functions, as well as tools to create custom versions
of them, ready to be used to analyze past operations data in the search for occurrences of
specific types of events in the past.
* **Prediction Engineering**: a flexible framework designed to apply labeling functions on
wind turbine operations data in a number of different ways to create labels for custom
Machine Learning problems.
* **Feature Engineering**: a guide to using Featuretools to apply automated feature engineerinig
to wind farm data.

# Install

## Requirements

**Zephyr** has been developed and runs on Python 3.6 and 3.7.

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid interfering
with other software installed in the system where you are trying to run **Zephyr**.

## Download and Install

**Zephyr** can be installed locally using [pip](https://pip.pypa.io/en/stable/) with
the following command:

```bash
pip install zephyr-ml
```

If you want to install from source or contribute to the project please read the
[Contributing Guide](CONTRIBUTING.rst).

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Zephyr**.

## 1. Loading the data

The first step we will be to use preprocessed data to create an EntitySet. Depending on the
type of data, we will either the `zephyr_ml.create_pidata_entityset` or `zephyr_ml.create_scada_entityset`
functions.

**NOTE**: if you cloned the **Zephyr** repository, you will find some demo data inside the
`notebooks/data` folder which has been preprocessed to fit the `create_entityset` data
requirements.

```python3
import os
import pandas as pd
from zephyr_ml import create_scada_entityset

data_path = 'notebooks/data'

data = {
  'turbines': pd.read_csv(os.path.join(data_path, 'turbines.csv')),
  'alarms': pd.read_csv(os.path.join(data_path, 'alarms.csv')),
  'work_orders': pd.read_csv(os.path.join(data_path, 'work_orders.csv')),
  'stoppages': pd.read_csv(os.path.join(data_path, 'stoppages.csv')),
  'notifications': pd.read_csv(os.path.join(data_path, 'notifications.csv')),
  'scada': pd.read_csv(os.path.join(data_path, 'scada.csv'))
}

scada_es = create_scada_entityset(data)
```

This will load the turbine, alarms, stoppages, work order, notifications, and SCADA data, and return it
as an EntitySet.

```
Entityset: SCADA data
  DataFrames:
    turbines [Rows: 1, Columns: 10]
    alarms [Rows: 2, Columns: 9]
    work_orders [Rows: 2, Columns: 20]
    stoppages [Rows: 2, Columns: 16]
    notifications [Rows: 2, Columns: 15]
    scada [Rows: 2, Columns: 5]
  Relationships:
    alarms.COD_ELEMENT -> turbines.COD_ELEMENT
    stoppages.COD_ELEMENT -> turbines.COD_ELEMENT
    work_orders.COD_ELEMENT -> turbines.COD_ELEMENT
    scada.COD_ELEMENT -> turbines.COD_ELEMENT
    notifications.COD_ORDER -> work_orders.COD_ORDER
```

## 2. Selecting a Labeling Function

The second step will be to choose an adequate **Labeling Function**.

We can see the list of available labeling functions using the `zephyr_ml.labeling.get_labeling_functions`
function.

```python3
from zephyr_ml import labeling

labeling.get_labeling_functions()
```

This will return us a dictionary with the name and a short description of each available
function.

```
{'brake_pad_presence': 'Calculates the total power loss over the data slice.',
 'converter_replacement_presence': 'Calculates the converter replacement presence.',
 'total_power_loss': 'Calculates the total power loss over the data slice.'}
 ```

In this case, we will choose the `total_power_loss` function, which calculates the total
amount of power lost over a slice of time.

## 3. Generate Target Times

Once we have loaded the data and the Labeling Function, we are ready to start using
the `zephyr_ml.generate_labels` function to generate a Target Times table.

```python3
from zephyr_ml import DataLabeler

data_labeler = DataLabeler(labeling.labeling_functions.total_power_loss)
target_times, metadata = data_labeler.generate_label_times(scada_es)
```

This will return us a `compose.LabelTimes` containing the three columns required to start
working on a Machine Learning problem: the turbine ID (COD_ELEMENT), the cutoff time (time) and the label.

```
   COD_ELEMENT       time    label
0            0 2022-01-01  45801.0
```

# What's Next?

If you want to continue learning about **Zephyr** and all its
features please have a look at the tutorials found inside the [notebooks folder](
https://github.com/signals-dev/zephyr/tree/main/notebooks).
