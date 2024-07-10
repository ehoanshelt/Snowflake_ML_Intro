import warnings
import pandas as pd
import os
import pickle
import yaml
import numpy as np
from snowflake.ml.modeling.preprocessing import OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.pipeline import Pipeline
from utils.snowflake import CoCam_SnowFlake
from utils.common import (clean_column_names)

warnings.simplefilter(action="ignore", category=UserWarning)

def featurize_data():
    titanic_df = pd.read_csv('data/raw/titanic.csv')
    titanic_df['FARE'] = titanic_df['FARE'].astype(float)
    titanic_df.drop(['ALIVE', 'DECK'], axis=1, inplace=True)
    titanic_df.dropna(subset="EMBARKED",inplace=True)
    cat_cols:list = titanic_df.select_dtypes(include="O").columns.to_list()

    cocam_sf = CoCam_SnowFlake()
    cocam_sf.connect()

    # NEed to add categorical pipeline
    # https://stackoverflow.com/questions/62409303/how-to-handle-missing-values-nan-in-categorical-data-when-using-scikit-learn-o

    # Need to add numic pipeline
    # simpleimputer will be mean and need to scale the inputs

    pipeline = Pipeline(
        steps=[
            (
            "SimpleImputer",
                SimpleImputer(
                    input_cols=cat_cols,
                    output_cols=cat_cols,
                    strategy='constant',
                    fill_value='missing',
                    drop_input_cols=True
                ),
            ),
            (
                "OneHotEncoder",
                OneHotEncoder(
                    input_cols=cat_cols,
                    output_cols=cat_cols,
                    drop_input_cols=True,
                    drop="first",
                    handle_unknown="ignore",
                ),
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
