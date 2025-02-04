from os import path
import pandas as pd
from zephyr_ml import create_scada_entityset

data_path = "notebooks/data"

data = {
    "turbines": pd.read_csv(path.join(data_path, "turbines.csv")),
    "alarms": pd.read_csv(path.join(data_path, "alarms.csv")),
    "work_orders": pd.read_csv(path.join(data_path, "work_orders.csv")),
    "stoppages": pd.read_csv(path.join(data_path, "stoppages.csv")),
    "notifications": pd.read_csv(path.join(data_path, "notifications.csv")),
    "scada": pd.read_csv(path.join(data_path, "scada.csv")),
}
scada_es = create_scada_entityset(data)

print(scada_es)
