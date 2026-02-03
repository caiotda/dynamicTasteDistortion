import argparse
import pickle
import os

import torch
from scipy.stats import expon

import numpy as np
import pandas as pd
from tqdm import tqdm

from dynamicTasteDistortion.simulation.simulator import Simulator


from dynamicTasteDistortion.simulationConstants import (
    input_size_to_file_name,
    MODEL_ARTIFACTS_PATH,
    SIMULATION_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets.")
    parser.add_argument(
        "--size",
        choices=["s", "m", "l"],
        required=True,
        help="Dataset size: s (1m), m (10m), l (20m).",
    )
    parser.add_argument(
        "--data",
        choices=["ml", "yelp", "steam"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; steam",
    )

    parser.add_argument(
        "--num_users",
        required=True,
        help="Number of users used to bootstrap clicks.",
    )
    args = parser.parse_args()
    num_users = int(args.num_users)
    data_type = args.data
    file_size = input_size_to_file_name[args.size]

    preference_matrix_path = (
        f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_oracle.pkl"
    )
    timestamp_output_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}_n_users={num_users}_avg_time_diff.csv"
    if os.path.exists(preference_matrix_path):
        print("Reading filled preference matrix...")
        oracle_df = pd.read_pickle(preference_matrix_path)
    else:
        print("Oracle matrix not found! Please run preference_model.py first.")
        return

    if os.path.exists(timestamp_output_path):
        print(f"Reading user timestamp behaviour from {timestamp_output_path}...")
        avg_std_time_diff_per_user = pd.read_csv(timestamp_output_path)
        userToExpDistribution = {
            user: expon(scale=row["median_timestamp_diff"])
            for user, row in avg_std_time_diff_per_user.iterrows()
        }
    else:
        print(
            "Timestamp behaviour file not found! Please run preference_model.py first."
        )
        return
    print(
        f"Bootstrapping clicks for {data_type}_{file_size} using {num_users} users..."
    )

    sim = Simulator(
        oracle_matrix=oracle_df,
        model=None,
        initial_date=0.0,
        user_timestamp_distribution=userToExpDistribution,
        bootstrapping_rounds=10,
    )
    print(f"Done! Saving bootstrapped clicks...")
    bootstrapped_clicks_path = f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_bootstrapped.pkl"
    click_matrix = sim.click_matrix
    with open(bootstrapped_clicks_path, "wb") as f:
        pickle.dump(click_matrix, f)
    print(f"Bootstrapped clicks saved to {bootstrapped_clicks_path}.")
