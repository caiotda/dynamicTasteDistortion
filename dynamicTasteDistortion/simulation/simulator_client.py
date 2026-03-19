import argparse
import pandas as pd
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
    get_oracle_matrix_path,
    get_timestamp_behavior_path,
    load_bootstrapped_clicks,
    load_pickle_artifact,
)

from dynamicTasteDistortion.dataset_loader import load_df


from scipy.stats import expon

from bprMf.bpr_mf import bprMFWithClickDebiasing, bprMf
from dynamicTasteDistortion.scripts.model_utils import MostPopularRecommender
from dynamicTasteDistortion.scripts.data_utils import standardize_ids
import yaml


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Load experiment config.")
    parser.add_argument(
        "--exp_file",
        default="experiment_config.yaml",
        help="YAML file with experiment configuration.",
    )

    args = parser.parse_args()

    with open(args.exp_file, "r") as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model", "bpr")
    data_type = cfg["data"]
    size = cfg["size"]
    file_size = input_size_to_file_name[size]

    exp_name = cfg.get("exp_name", "default_experiment")
    calibration_type = cfg.get("calibrate", None)
    calibration_type = (
        str.lower(calibration_type) if calibration_type is not None else None
    )
    assert calibration_type in [
        None,
        "rating",
        "constant",
        "linear_time",
        "exponential_time",
    ], "Invalid calibration type specified in config."

    use_oracle_matrix = True if cfg.get("use_oracle_matrix", "n") == "y" else False

    rounds = int(cfg["rounds"])
    num_rounds_per_eval = int(cfg["num_rounds_per_eval"])
    num_users = int(cfg["num_users"])

    preference_update_rate = float(cfg.get("preference_update_rate"), 0)

    timestamp_distribution = pd.read_csv(
        get_timestamp_behavior_path(data_type, file_size, num_users)
    )
    oracle_matrix = load_pickle_artifact(
        get_oracle_matrix_path(data_type, file_size, num_users)
    )
    bootstrapped_df = load_bootstrapped_clicks(data_type, file_size, num_users)

    n_users = oracle_matrix[USER_COL].max() + 1
    n_items = oracle_matrix[ITEM_COL].max() + 1

    if model_type == "bpr":
        model = bprMFWithClickDebiasing(
            num_users=n_users,
            num_items=n_items,
            factors=30,
            n_epochs=1,
            reg_lambda=5e-4,
            dev=device,
            lr=1e-3,
        )
    elif model_type == "bpr_classic":
        model = bprMf(
            num_users=n_users,
            num_items=n_items,
            factors=30,
            n_epochs=1,
            reg_lambda=5e-4,
            dev=device,
            lr=1e-3,
        )
    elif model_type == "most_popular":
        print(f"Loading {data_type}_{file_size} dataset to fit Most Popular model...")
        df = load_df(data_type, size)
        processed_df, _, _ = standardize_ids(df)
        model = MostPopularRecommender(processed_df)
    else:
        model = None

    userToExpDistribution = {
        user: expon(scale=row["median_timestamp_diff"])
        for user, row in timestamp_distribution.iterrows()
    }
    base_artifacts_path = (
        Path(RESULTS_PATH)
        / f"{data_type}_{file_size}"
        / "simulated"
        / f"exp={exp_name}"
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
        ignore_oracle_matrix=not use_oracle_matrix,
        calibration_type=calibration_type,
        preference_update_rate=preference_update_rate
    )
    simulated_df, maces, kl_divs = sim.simulate(
        L=num_rounds_per_eval, rounds=rounds, k=20
    )

    print(f"Done! Saving simulated interactions...")
    simulated_df.to_pickle(base_artifacts_path / "simulated_interactions.pkl")

    with open(base_artifacts_path / "maces.pkl", "wb") as f:
        pickle.dump(maces, f)

    with open(base_artifacts_path / "kl_divs.pkl", "wb") as f:
        pickle.dump(kl_divs, f)
