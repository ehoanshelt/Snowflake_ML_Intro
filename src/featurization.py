import warnings
import pandas as pd
import os
import pickle
from snowflake.ml.modeling.preprocessing import OneHotEncoder
from snowflake.ml.modeling.pipeline import Pipeline
from utils.snowflake import CoCam_SnowFlake
from utils.common import (clean_column_names)

warnings.simplefilter(action="ignore", category=UserWarning)


def featurize_data():
    titanic_df = pd.read_csv('data/raw/titanic.csv')
    titanic_df.dropna(inplace=True)
    titanic_df['FARE'] = titanic_df['FARE'].astype(float)
    cat_cols = titanic_df.select_dtypes(include=['object']).columns

    #print("---- Found Categorical Column" + cat_cols )

    cocam_sf = CoCam_SnowFlake()
    cocam_sf.connect()

    pipeline = Pipeline(
        steps=[
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

    print("---- Running Pipeline")

    pipeline.fit(titanic_df)
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
