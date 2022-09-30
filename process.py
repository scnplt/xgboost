from sklearn.preprocessing import LabelEncoder, RobustScaler
from constant import *

import pandas as pd
import numpy as np
import pickle
import os

def process_data(dataframe):
    new_df = fill_columns_with_mod(dataframe)
    new_df = convert_dtypes_to_numerical_if_needed(new_df)
    new_df = convert_dtypes_to_categorical_if_needed(new_df)

    categorical_columns = get_data_from_file(
        path = CATEGORICAL_COLUMNS_PATH,
        create_data_callback = lambda: [col for col in new_df.columns if new_df[col].dtypes == "category"])

    numerical_columns = get_data_from_file(
        path = NUMERICAL_COLUMNS_PATH,
        create_data_callback = lambda: new_df.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    )

    encoders = get_data_from_file(
        path = LABEL_ENCODERS_PATH,
        create_data_callback=lambda: create_label_encoders_dict(new_df, categorical_columns)
    )

    for col, encoder in encoders.items():
        new_df[col] = encoder.transform(new_df[col])

    robust_scalers = get_data_from_file(
        path = ROBUST_SCALERS_PATH,
        create_data_callback= lambda: create_robust_scalers_dict(new_df, numerical_columns)
    )

    for col, scaler in robust_scalers.items():
        new_df[col] = scaler.transform(new_df[col].to_numpy().reshape(-1, 1))

    return new_df
    
def get_data_from_file(path, create_file_if_not_exist=True, create_data_callback=None):
    if (os.path.isfile(path)): return pickle.load(open(path, "rb"))

    data = create_data_callback()
    if (create_file_if_not_exist):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    return data

def create_label_encoders_dict(dataframe, categorical_columns):
    return {col: LabelEncoder().fit(dataframe[col]) for col in categorical_columns}

def create_robust_scalers_dict(dataframe, columns):
    return {col: RobustScaler().fit(dataframe[col].to_numpy().reshape(-1, 1)) for col in columns}
    
def fill_columns_with_mod(dataframe):
    dataframe = dataframe.replace(' ', np.nan)
    for col in dataframe.columns:
        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
    return dataframe

def convert_dtypes_to_numerical_if_needed(dataframe):
    for col in dataframe.columns:
        try:
            dataframe[col] = dataframe[col].astype('float64')
        except Exception as e:
            continue
    return dataframe

def convert_dtypes_to_categorical_if_needed(dataframe):
    categorical_dtypes = ["object", "category", "bool"]
    for col in dataframe.columns:
        if (dataframe[col].dtypes in categorical_dtypes):
            dataframe[col] = dataframe[col].astype("category")
    return dataframe