import warnings
import pandas as pd
import os
import pickle
import yaml
import numpy as np
from snowflake.ml.modeling.preprocessing import OneHotEncoder, StandardScaler
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.compose import ColumnTransformer
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
    num_cols = ["AGE", "FARE"]

    cocam_sf = CoCam_SnowFlake()
    cocam_sf.connect()

    # NEed to add categorical pipeline
    # https://stackoverflow.com/questions/62409303/how-to-handle-missing-values-nan-in-categorical-data-when-using-scikit-learn-o

    # Need to add numic pipeline
    # simpleimputer will be mean and need to scale the inputs

    cat_transformers = Pipeline(
        steps=[
            (
            "SimpleImputer",
                SimpleImputer(
                    strategy='constant',
                    fill_value='missing',
                    drop_input_cols=True
                ),
            ),
            (
                "OneHotEncoder",
                OneHotEncoder(
                    drop_input_cols=True,
                    drop="first",
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    numeric_transformers = Pipeline(
            steps=[
                (
                "SimpleImputer",
                    SimpleImputer(
                        strategy='mean',
                        drop_input_cols=True
                    ),
                ),
                (
                    "StandardScaler",
                    StandardScaler(
                        drop_input_cols=True
                    ),
                )
            ]
        )
    
    preprocessor = ColumnTransformer(
                        input_cols = cat_cols + num_cols,
                        transformers=[
                            ('cat', cat_transformers,cat_cols),
                            ('num', numeric_transformers, num_cols)
                        ],
                        output_cols = cat_cols + num_cols,
                )

    print("---- Fitting Pipeline")
    preprocessor.fit(titanic_df)

    print("---- Transforming Data")
    featurized_df = preprocessor.transform(titanic_df)

    featurized_df.columns = clean_column_names(featurized_df)

    print("---- Writing Featurized Dataset")

    os.makedirs('data/featurized', exist_ok=True)
    featurized_df.to_csv("data/featurized/titanic.csv", index=False, mode='w+')

    print("---- Writing Pipeline Pickle File")

    os.makedirs('models/featurized', exist_ok=True)
    pickle.dump(preprocessor, open('models/featurized/featurized.pkl', 'wb'))

    print("---- Successfully Featurized Data")



if __name__ == "__main__":
    featurize_data()
