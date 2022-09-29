from constant import UNIQUE_VALUES_IN_COLUMNS, FAKE_DATA_FOLDER_PATH
from process import *

import pandas as pd
import random as rnd
import time

def elapsed_time(program):
    start = time.time()
    result = program()
    end = time.time()
    return end - start, result

def fake_data_generator(df, row_size):
    df = fill_columns_with_mod(dataframe = df)
    df = convert_dtypes_to_numerical_if_needed(df)
    df = convert_dtypes_to_categorical_if_needed(df)

    uniq_values = get_data_from_file(
        path = UNIQUE_VALUES_IN_COLUMNS, 
        create_data_callback = lambda: [df[col].unique() if df[col].dtype == "category" else list(range(int(df[col].min()), int(df[col].max()))) for col in df.columns if col != "Churn"])

    data = [[rnd.choice(values) for values in uniq_values] for _ in range(row_size)]

    new_df = pd.DataFrame(data=data, columns=[col for col in df.columns if col != "Churn"])
    new_df["TotalCharges"] = new_df["MonthlyCharges"]
    new_df.to_csv(f"{FAKE_DATA_FOLDER_PATH}\\fake_data_{row_size}.csv", index=False)
