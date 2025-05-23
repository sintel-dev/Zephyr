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
| ![][Slack Logo] **[Community]**     | Join our Slack Workspace for announcements and discussions.          |

[Website]: https://sintel.dev/
[Documentation]: https://dtail.gitbook.io/zephyr/
[Repository]: https://github.com/sintel-dev/Zephyr
[Tutorials]: https://github.com/sintel-dev/Zephyr/blob/master/notebooks
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

- **EntitySet creation**: tools designed to represent wind farm data and the relationship
  between different tables. We have functions to create EntitySets for datasets with PI data
  and datasets using SCADA data.
- **Labeling Functions**: a collection of functions, as well as tools to create custom versions
  of them, ready to be used to analyze past operations data in the search for occurrences of
  specific types of events in the past.
- **Prediction Engineering**: a flexible framework designed to apply labeling functions on
  wind turbine operations data in a number of different ways to create labels for custom
  Machine Learning problems.
- **Feature Engineering**: a guide to using Featuretools to apply automated feature engineerinig
  to wind farm data.

# Install

## Requirements

**Zephyr** has been developed and runs on Python 3.8, 3.9, 3.10, 3.11 and 3.12.

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid interfering
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
getting started with **Zephyr**. For more detailed examples, please refer to the tutorial notebooks in the `notebooks` directory:

- `feature_engineering.ipynb`: Learn how to create EntitySets and perform feature engineering
- `modeling.ipynb`: Learn how to train and evaluate models
- `visualization.ipynb`: Learn how to visualize your data and results

## 1. Loading the data

The first step will be to use preprocessed data to create an EntitySet. Depending on the
type of data, we will use either the `generate_entityset` function with `es_type="pidata"`, `es_type="scada"` or `es_type="vibrations"`.

**NOTE**: if you cloned the **Zephyr** repository, you will find some demo data inside the
`notebooks/data` folder which has been preprocessed to fit the data requirements.

```python
import os
import pandas as pd
from zephyr_ml import Zephyr

data_path = 'notebooks/data'

data = {
    'turbines': pd.read_csv(os.path.join(data_path, 'turbines.csv')),
    'alarms': pd.read_csv(os.path.join(data_path, 'alarms.csv')),
    'work_orders': pd.read_csv(os.path.join(data_path, 'work_orders.csv')),
    'stoppages': pd.read_csv(os.path.join(data_path, 'stoppages.csv')),
    'notifications': pd.read_csv(os.path.join(data_path, 'notifications.csv')),
    'scada': pd.read_csv(os.path.join(data_path, 'scada.csv'))
}

zephyr = Zephyr()
scada_es = zephyr.generate_entityset(data, es_type="scada")
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

The second step will be to choose an adequate **Labeling Function**. We can see the list of available labeling functions using the `GET_LABELING_FUNCTIONS` method.

```python
labeling_functions = zephyr.GET_LABELING_FUNCTIONS()
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
the `generate_label_times` function to generate a Target Times table.

```python
target_times, metadata = zephyr.generate_label_times(
    labeling_fn="total_power_loss",  # or any other labeling function name
    num_samples=10,
    gap="20d"
)
```

This will return us a `compose.LabelTimes` containing the three columns required to start
working on a Machine Learning problem: the turbine ID (COD_ELEMENT), the cutoff time (time) and the label.

```
   COD_ELEMENT       time    label
0            0 2022-01-01  45801.0
```

## 4. Feature Engineering

Using EntitySets and LabelTimes allows us to easily use Featuretools for automatic feature generation.

```python
feature_matrix, features, _ = zephyr.generate_feature_matrix(
    target_dataframe_name="turbines",
    cutoff_time_in_index=True,
    agg_primitives=["count", "sum", "max"],
    max_features = 20,
    verbose=True
)
```

Then we get a list of features and the computed `feature_matrix`.

```
                       TURBINE_PI_ID TURBINE_LOCAL_ID TURBINE_SAP_COD DES_CORE_ELEMENT      SITE DES_CORE_PLANT  ... MODE(alarms.COD_STATUS) MODE(alarms.DES_NAME)  MODE(alarms.DES_TITLE)  NUM_UNIQUE(alarms.COD_ALARM)  NUM_UNIQUE(alarms.COD_ALARM_INT)    label
COD_ELEMENT time                                                                                                 ...
0           2022-01-01          TA00               A0          LOC000              T00  LOCATION            LOC  ...                  Alarm1                Alarm1  Description of alarm 1                             1                                 1  45801.0

[1 rows x 21 columns]
```

## 5. Modeling

Once we have the feature matrix, we can train a model using the Zephyr interface. First, we need to prepare our dataset for training by creating a train-test split.

```python
X_train, X_test, y_train, y_test = zephyr.generate_train_test_split(
    test_size=0.2,
    random_state=42
)
```

In this example, we will use an 'xgb' regression pipeline to predict total power loss. To train the pipeline, we simply call the `fit_pipeline` method.

```python
zephyr.fit_pipeline(
    pipeline="xgb_regressor",
    pipeline_hyperparameters=None,

)
```

After it finished training, we can make predictions using `predict`

```python
y_pred = zephyr.predict(X_test)
```

We can also use `evaluate` to obtain the performance of the pipeline.

```python
results = zephyr.evaluate()
```

# What's Next?

If you want to continue learning about **Zephyr** and all its
features please have a look at the tutorials found inside the [notebooks folder](https://github.com/signals-dev/zephyr/tree/main/notebooks).
