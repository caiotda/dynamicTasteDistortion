import argparse
import torch
import pickle
from pathlib import Path

from dynamicTasteDistortion.simulationConstants import (
    ITEM_COL,
    USER_COL,
    input_size_to_file_name,
    RESULTS_PATH,
)
from dynamicTasteDistortion.simulation.simulator import Simulator

from dynamicTasteDistortion.ioUtils import (
    load_bootstrapped_clicks,
    load_time_diff_df,
    load_oracle_matrix,
)


from scipy.stats import expon

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
        "--num_users",
        required=True,
        help="Number of users used in the simulation.",
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
    num_users = int(args.num_users)

    timestamp_distribution = load_time_diff_df(data_type, file_size)
    oracle_matrix = load_oracle_matrix(data_type, file_size)
    bootstrapped_df = load_bootstrapped_clicks(data_type, file_size)

    n_users = oracle_matrix[USER_COL].max() + 1
    n_items = oracle_matrix[ITEM_COL].max() + 1

    if num_users is not None:
        candidates = timestamp_distribution.index.tolist()
        idx = torch.randperm(len(candidates))[:num_users]
        users = [candidates[i] for i in idx.tolist()]
        timestamp_distribution = timestamp_distribution.iloc[users]
        oracle_matrix = oracle_matrix[oracle_matrix[USER_COL].isin(users)]
        bootstrapped_df = bootstrapped_df[bootstrapped_df[USER_COL].isin(users)]

    model = bprMFWithClickDebiasing(
        num_users=n_users,
        num_items=n_items,
        factors=30,
        n_epochs=1,
        reg_lambda=5e-4,
        dev=device,
        lr=1e-3,
    )

    userToExpDistribution = {
        user: expon(scale=row["median_timestamp_diff"])
        for user, row in timestamp_distribution.iterrows()
    }
    base_artifacts_path = (
        Path(RESULTS_PATH)
        / f"{data_type}_{file_size}"
        / "simulated"
        / f"rounds={rounds}"
        / f"users={num_users}"
        / f"eval_every={num_rounds_per_eval}"
    )

    sim = Simulator(
        oracle_matrix=oracle_matrix,
        model=model,
        initial_date=0.0,
        user_timestamp_distribution=userToExpDistribution,
        bootstrapped_df=bootstrapped_df,
        base_artifacts_path=base_artifacts_path,
    )
    simulated_df, maces, kl_divs = sim.simulate(L=num_rounds_per_eval, rounds=rounds)

    print(f"Done! Saving simulated interactions...")
    simulated_df.to_pickle(base_artifacts_path / "simulated_interactions.pkl")

    with open(base_artifacts_path / "maces.pkl", "wb") as f:
        pickle.dump(maces, f)

    with open(base_artifacts_path / "kl_divs.pkl", "wb") as f:
        pickle.dump(kl_divs, f)
