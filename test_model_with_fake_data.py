import pandas as pd
import pickle
import time

from process import process_data

def elapsed_time(program):
    start = time.time()
    result = program()
    end = time.time()
    return end-start, result

def test_with_fake_data():
    model = pickle.load(open("pickle_files\\model.pkl", "rb"))

    time, df = elapsed_time(lambda: pd.read_csv("csv\\fake_data.csv"))
    print("Time of loading data from csv file: %.2fs" % time)

    time, new_df = elapsed_time(lambda: process_data(df))
    print("Time of processing data: %.2fs" % time)

    time, result = elapsed_time(lambda: model.predict(new_df))
    print("Time of prediction: %.2fs" % time)
    print(f"Result list: {result} - Length of result: {len(result)}")

test_with_fake_data()