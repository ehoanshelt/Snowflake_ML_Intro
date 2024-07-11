import warnings
import pandas as pd
import os
import pickle
import yaml
import numpy as np
from snowflake.ml.modeling.preprocessing import OneHotEncoder, StandardScaler
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.pipeline import Pipeline
from utils.snowflake import CoCam_SnowFlake
from utils.common import (clean_column_names)

warnings.simplefilter(action="ignore", category=UserWarning)

def featurize_data():
    titanic_df = pd.read_csv('data/raw/titanic.csv')
    titanic_df['FARE'] = titanic_df['FARE'].astype(float)
    titanic_df.dropna(subset="EMBARKED",inplace=True)
    titanic_df["IS_CHILD"] = titanic_df["WHO"].apply(lambda x: True if x == 'child' else False)
    titanic_df.drop(['ALIVE', 'DECK', 'ADULT_MALE', 'WHO'], axis=1, inplace=True)
    cat_cols:list = titanic_df.select_dtypes(include="O").columns
    num_cols:list = titanic_df.drop('SURVIVED', axis=1).select_dtypes(include=['int64', 'float64']).columns

    cocam_sf = CoCam_SnowFlake()
    print("---- Connecting to Snowflake")
    cocam_sf.connect()

    pipeline = Pipeline(
        [
            (
            'categorical_imputer',
            SimpleImputer( 
                input_cols = cat_cols,
                strategy='most_frequent',
                output_cols=cat_cols
                )
            ),
            (
            "OneHotEncoder",
            OneHotEncoder(
                input_cols= cat_cols,
                drop="first",
                handle_unknown="ignore",
                drop_input_cols = True,
                output_cols=cat_cols
            )
            ),
            (
            "numeric_imputer",
                SimpleImputer(
                    input_cols = num_cols,
                    strategy='mean',
                    output_cols=num_cols
                )
            ),
            (
            "StandardScaler",
                StandardScaler(
                    input_cols = num_cols,
                    drop_input_cols = True,
                    output_cols=num_cols
                )
            )
        ]
    )

    print("---- Fitting Pipeline")
    pipeline.fit(titanic_df)

    print("---- Transforming Data")
    featurized_df = pipeline.transform(titanic_df)

    featurized_df.columns = clean_column_names(featurized_df)

    print("---- Writing Featurized Dataset")

    os.makedirs('data/featurized', exist_ok=True)
    featurized_df.to_csv("data/featurized/titanic.csv", index=False, mode='w+')

    print("---- Writing Pipeline Pickle File")

    os.makedirs('models/featurized', exist_ok=True)
    pickle.dump(pipeline, open('models/featurized/featurized.pkl', 'wb'))

    print("---- Successfully Featurized Data")



if __name__ == "__main__":
    featurize_data()
