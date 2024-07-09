import os
import yaml
from dotenv import load_dotenv
import pandas as pd
import snowflake.snowpark.functions as F
from utils.snowflake import CoCam_SnowFlake


def load_data():
    load_dotenv()
    cocam_sf = CoCam_SnowFlake(os.getenv('SNOWFLAKE_DATABASE'), os.getenv('SNOWFLAKE_SCHEMA'))
    cocam_sf.connect()
    cocam_sf.create_snowflake_stage("TITANIC_DATA")

    model_version = yaml.safe_load(open("params.yaml"))["model"]["version"]
    source_data = yaml.safe_load(open("params.yaml"))["model"]["source_data"]

    titanic = pd.read_csv(source_data)
    titanic.columns = [c.upper() for c in titanic.columns]

    os.makedirs('data/raw', exist_ok=True)
    titanic.to_csv("data/raw/titanic.csv", index=False, mode='w+')


if __name__ == "__main__":
    load_data()
    print("Successfully loaded Data")