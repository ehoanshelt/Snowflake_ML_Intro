import warnings
import pandas as pd
import os
import pickle
from dotenv import load_dotenv
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.preprocessing import OneHotEncoder
from snowflake.ml.modeling.pipeline import Pipeline
from utils.common import (count_all_nulls)
from utils.snowflake import CoCam_SnowFlake

warnings.simplefilter(action="ignore", category=UserWarning)


def featurize_data():
    titanic_df = pd.read_csv('data/raw/titanic.csv')
    columns_with_nulls = count_all_nulls(titanic_df)
    titanic_df.drop(columns_with_nulls, axis=1, inplace=True)
    titanic_df['FARE'] = titanic_df['FARE'].astype(float)
    cat_cols = titanic_df.select_dtypes(include=['object'])

    cocam_sf = CoCam_SnowFlake()
    cocam_sf.connect()

    pipeline = Pipeline(
        steps=[
            (
                "SimpleImputer",
                SimpleImputer(
                    input_cols=cat_cols,
                    output_cols=cat_cols,
                    strategy="most_frequent",
                    drop_input_cols=True,
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

    pipeline.fit(titanic_df).transform(titanic_df)

    os.makedirs('data/featurized', exist_ok=True)
    titanic_df.to_csv("data/featurized/titanic.csv", index=False, mode='w+')

    os.makedirs('models/featurized', exist_ok=True)
    pickle.dump(pipeline, open('models/featurized/featurized.pkl', 'wb'))




if __name__ == "__main__":
    featurize_data()
    print("Successfully loaded Data")