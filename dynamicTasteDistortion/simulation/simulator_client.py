import argparse
import torch

from dynamicTasteDistortion.simulationConstants import (
    ITEM_COL,
    USER_COL,
    input_size_to_file_name,
)
from dynamicTasteDistortion.simulation.simulator import Simulator

from dynamicTasteDistortion.ioUtils import (
    load_time_diff_df,
    load_oracle_matrix,
)
from bprMf.bpr_mf import bprMFWithClickDebiasing


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "--rounds",
        required=True,
        help="Number of simulation rounds.",
    )

    parser.add_argument(
        "--num_rounds_per_eval",
        required=True,
        help="Number of rounds before triggering retraining and MACE measuring",
    )

    args = parser.parse_args()
    data_type = args.data
    size = args.size
    file_size = input_size_to_file_name[size]
    rounds = int(args.rounds)
    num_rounds_per_eval = int(args.num_rounds_per_eval)

    timestamp_distribution = load_time_diff_df(data_type, file_size)
    oracle_matrix = load_oracle_matrix(data_type, file_size)

    n_users = oracle_matrix[USER_COL].max() + 1
    n_items = oracle_matrix[ITEM_COL].max() + 1

    model = bprMFWithClickDebiasing(
        num_users=n_users, num_items=n_items, factors=30, n_epochs=1, reg_lambda=5e-4
    ).to(device)

    sim = Simulator(
        oracle_matrix=oracle_matrix,
        model=model,
        initial_date=0.0,
        user_timestamp_distribution=timestamp_distribution,
    )
