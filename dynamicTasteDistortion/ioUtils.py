import ast
from dynamicTasteDistortion.scripts.metrics_utils import remove_outliers
from dynamicTasteDistortion.scripts.model_utils import (
    choose_best_model,
)

from dynamicTasteDistortion.scripts.bootstrapping_utils import (
    fill_out_matrix,
    get_timestamp_behavior,
)
from dynamicTasteDistortion.simulationConstants import (
    MODEL_ARTIFACTS_PATH,
    SIMULATION_PATH,
    RESULTS_PATH,
    input_size_to_file_name,
)

import os
from pathlib import Path

import pandas as pd
import pickle

import yaml


def read_experiment(exp_file):
    with open(exp_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def read_metrics(cfg_file, should_remove_outliers=False):
    data_type = cfg_file["data"]
    size = cfg_file["size"]
    file_size = input_size_to_file_name[size]

    rounds = int(cfg_file["rounds"])
    num_rounds_per_eval = int(cfg_file["num_rounds_per_eval"])
    num_users = int(cfg_file["num_users"])

    exp_name = cfg_file.get("exp_name", "default_experiment")

    base_artifacts_path = (
        Path(RESULTS_PATH)
        / f"{data_type}_{file_size}"
        / "simulated"
        / f"exp={exp_name}"
        / f"rounds={rounds}"
        / f"users={num_users}"
        / f"eval_every={num_rounds_per_eval}"
    )
    # Read maces pickle file
    file_name = base_artifacts_path / "maces.pkl"
    maces = pd.read_pickle(file_name)
    
    # Read kl divs pickle file
    file_name = base_artifacts_path / "kl_divs.pkl"
    kl_divs = pd.read_pickle(file_name)

    # Read MAP pickle file
    # TODO: salvei esse arquivo com o nome errado. Vou ter que arrumar
    # na mão, rerodar, e depois ajustar aqui e no simulator_client
    file_name = base_artifacts_path / "mace_at_k.pkl"
    map_k = pd.read_pickle(file_name)

    # Read catalog_coverage pickle file
    file_name = base_artifacts_path / "catalog_coverage.pkl"
    catalog_coverage = pd.read_pickle(file_name)

    # Read MRR
    file_name = base_artifacts_path / "mrrs.pkl"
    mrr = pd.read_pickle(file_name)
    if should_remove_outliers:
        maces = remove_outliers(maces)
        kl_divs = remove_outliers(kl_divs)
        map_k = remove_outliers(map_k)
        catalog_coverage = remove_outliers(catalog_coverage)
        mrr = remove_outliers(mrr)
    return maces, kl_divs, map_k, catalog_coverage, mrr


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


def save_pickle_artifact(artifact, path):
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def get_base_model_path(data_type, file_size, num_users, model_type):
    return f"{MODEL_ARTIFACTS_PATH}/{model_type}_{data_type}_{file_size}_n_users={num_users}"


def get_model_path(data_type, file_size, num_users, model_type):
    base_path = get_base_model_path(data_type, file_size, num_users, model_type)
    str_path = f"{base_path}_model.pkl"
    return str_path


def get_best_params_path(data_type, file_size, num_users, model_type):
    base_path = get_base_model_path(data_type, file_size, num_users, model_type)
    str_path = f"{base_path}_params.pkl"
    return str_path


def get_cv_results_path(data_type, file_size, num_users, model_type):
    base_path = get_base_model_path(data_type, file_size, num_users, model_type)
    str_path = f"{base_path}_cv_results.csv"
    return str_path


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
    oracle_model_path = (
        f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/oracle_model.pkl"
    )

    if os.path.exists(oracle_model_path):
        print(
            f"Oracle model trained on {data_type}_{size} found!Skipping model selection"
        )
        oracle_model_artifact = load_pickle_artifact(oracle_model_path)

        model_class = oracle_model_artifact["model_class"]
        model_params = oracle_model_artifact["params"]
        oracle_model = model_class(**model_params)
    else:
        os.makedirs(f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/", exist_ok=True)
        print("Starting model selection...")
        oracle_model, f1_results = choose_best_model(df)
        save_pickle_artifact(oracle_model, oracle_model_path)
        destination_dir = f"{RESULTS_PATH}/{data_type}_{size}"
        os.makedirs(destination_dir, exist_ok=True)
        f1_results.to_pickle(f"{destination_dir}/oracle_model_f1_results.pkl")
        print(f"Saving oracle model f1 results to {destination_dir}/oracle_model_f1_results.pkl")
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
