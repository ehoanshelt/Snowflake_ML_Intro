import os
import yaml
from dotenv import load_dotenv
import pandas as pd
import snowflake.snowpark.functions as F
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

class CoCam_SnowFlake():
    def __init__(self, SF_DATABASE:str,SF_SCHEMA:str ) -> None:
        self.database = SF_DATABASE
        self.schema = SF_SCHEMA
    
    def __str__(self):
        return f"Use .connect function to connect to {self.database}.{self.schema}"


    def connect(self) -> None:
        load_dotenv()
        session = Session.builder.configs(SnowflakeLoginOptions()).getOrCreate()
        session.sql("USE " + self.database).collect()
        session.sql("CREATE SCHEMA IF NOT EXISTS " + self.schema).collect()
        session.sql("USE SCHEMA " + self.schema ).collect()
        self.session = session
        return 

    def create_snowflake_stage(self, STAGE_NAME:str):
        self.session.sql("CREATE STAGE if not exists " + STAGE_NAME).collect()
        self.stage = STAGE_NAME
        return