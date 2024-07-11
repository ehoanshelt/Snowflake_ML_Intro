import os
import yaml
import pandas as pd
import snowflake.snowpark.functions as F
from utils.snowflake import CoCam_SnowFlake


def load_data():
    cocam_sf = CoCam_SnowFlake()
    cocam_sf.connect()

    source_data = yaml.safe_load(open("params.yaml"))["model"]["source_data"]

    titanic = pd.read_csv(source_data)
    titanic.columns = [c.upper() for c in titanic.columns]

    os.makedirs('data/raw', exist_ok=True)
    titanic.to_csv("data/raw/titanic.csv", index=False, mode='w+')


if __name__ == "__main__":
    load_data()
    print("Successfully loaded Data")