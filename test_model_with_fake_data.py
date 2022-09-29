from process import process_data
from constant import MODEL_PATH, FAKE_DATA_PATH_10_000_000, FAKE_DATA_PATH_1_000_000
from utils import elapsed_time

import pandas as pd
import pickle

def test_with_fake_data(data_path):
    model = pickle.load(open(MODEL_PATH, "rb"))

    time, df = elapsed_time(lambda: pd.read_csv(data_path))
    print("Time of loading data from csv file: %.2fs" % time)

    time, new_df = elapsed_time(lambda: process_data(df))
    print("Time of processing data: %.2fs" % time)

    time, result = elapsed_time(lambda: model.predict(new_df))
    print("Time of prediction: %.2fs" % time)
    print(f"Result list: {result} - Length of result: {len(result)}")
    print(f"count '0': {list(result).count(0)} - count '1': {list(result).count(1)}")

print("1M Data")
test_with_fake_data(data_path = FAKE_DATA_PATH_1_000_000)

print("\n10M Data")
test_with_fake_data(data_path = FAKE_DATA_PATH_10_000_000)
