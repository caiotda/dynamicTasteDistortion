from dynamicTasteDistortion.preference_model import (
    choose_best_model,
    # \/ Dependencia circular.
    fill_out_matrix,
    get_timestamp_behavior,
)
from dynamicTasteDistortion.simulationConstants import (
    MODEL_ARTIFACTS_PATH,
    SIMULATION_PATH,
)

import os


import pandas as pd
import pickle


def load_bootstrapped_clicks(data_type, size, num_users):
    output_path = (
        f"{SIMULATION_PATH}/{data_type}_{size}_n_users={num_users}_bootstrapped.pkl"
    )
    with open(output_path, "rb") as f:
        bootstrapped_clicks = pickle.load(f)
    return bootstrapped_clicks


def load_pickle_artifact(base_path):
    with open(base_path, "rb") as f:
        pickle_artifact = pickle.load(f)
    return pickle_artifact


def get_oracle_matrix_path(data_type, file_size, num_users):
    return f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_oracle.pkl"


def get_timestamp_behavior_path(data_type, file_size, num_users):
    return f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}_n_users={num_users}_avg_time_diff.csv"


def get_or_create_oracle_matrix(oracle_model, df, data_type, file_size, users):
    num_users = len(users)
    oracle_output_path = get_oracle_matrix_path(data_type, file_size, num_users)
    if os.path.exists(oracle_output_path):
        print(
            f"Filled oracle matrix for {data_type}_{file_size} that uses {num_users} users already exists! Skipping matrix filling."
        )
        filled_oracle_matrix = load_pickle_artifact(oracle_output_path)
    else:
        print("Filling up rating matrix...")
        filled_oracle_matrix = fill_out_matrix(
            base_df=df, model=oracle_model, user_sample=users
        )
        print(f"Writing filled out matrix to {oracle_output_path}")
    filled_oracle_matrix.to_pickle(oracle_output_path)
    return filled_oracle_matrix


def get_or_create_oracle_model_artifacts(df, data_type, size):
    oracle_model_params_path = (
        f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/oracle_model_params.pkl"
    )

    if os.path.exists(oracle_model_params_path):
        print(
            f"Oracle model trained on {data_type}_{size} found!Skipping model selection"
        )
        oracle_model_artifact = load_pickle_artifact(oracle_model_params_path)

        model_class = oracle_model_artifact["model_class"]
        model_params = oracle_model_artifact["params"]
        oracle_model = model_class(**model_params)
    else:
        print("Starting model selection...")
        oracle_model = choose_best_model(df, f"{data_type}_{size}")
        # TODO: falta persistir os parametros.

    return oracle_model


def get_or_create_time_diff_df(df, data_type, size, users):
    num_users = len(users)
    timestamp_path = get_timestamp_behavior_path(data_type, size, num_users)
    if os.path.exists(timestamp_path):
        print(
            f"Timestamp behavior for {data_type}_{size} that uses {num_users} users already exists! Skipping timestamp behavior calculation."
        )
        avg_std_time_diff_per_user = pd.read_csv(timestamp_path)
    else:
        avg_std_time_diff_per_user = get_timestamp_behavior(base_df=df, sample=users)

        if len(avg_std_time_diff_per_user) == 0:
            print("Timestamp df is emtpy! Please check get_timestamp_behavior func")
            return
        print(f"timestamp diff: {avg_std_time_diff_per_user}")
        print(f"Writing timestamp behavior per user to {timestamp_path}")
        avg_std_time_diff_per_user.to_csv(
            timestamp_path,
            index=False,
        )
