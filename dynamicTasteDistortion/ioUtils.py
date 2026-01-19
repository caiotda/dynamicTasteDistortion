from dynamicTasteDistortion.simulationConstants import (
    MODEL_ARTIFACTS_PATH,
    SIMULATION_PATH,
)


import pandas as pd
import pickle


def load_time_diff_df(data_type, size):
    timestamp_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/avg_time_diff.csv"
    time_diff_df = pd.read_csv(timestamp_path)
    return time_diff_df


def load_oracle_matrix(data_type, size):
    output_path = f"{SIMULATION_PATH}/{data_type}_{size}_oracle.pkl"
    with open(output_path, "rb") as f:
        filled_oracle_matrix = pickle.load(f)
    return filled_oracle_matrix


def load_bootstrapped_clicks(data_type, size):
    output_path = f"{SIMULATION_PATH}/{data_type}_{size}_bootstrapped.pkl"
    with open(output_path, "rb") as f:
        bootstrapped_clicks = pickle.load(f)
    return bootstrapped_clicks
