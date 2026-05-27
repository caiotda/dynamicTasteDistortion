import argparse
import os
import pandas as pd

from dynamicTasteDistortion.simulation.simulator import Simulator

from dynamicTasteDistortion.ioUtils import (
    get_experiment_artifacts_path,
    get_oracle_matrix_path,
    get_timestamp_behavior_path,
    instantiate_model,
    load_bootstrapped_clicks,
    load_pickle_artifact,
    save_pickle_artifact,
    extract_experiment_configuration,
)

from scipy.stats import expon


import yaml


def main():
    parser = argparse.ArgumentParser(description="Load experiment config.")
    parser.add_argument(
        "--exp_file",
        default="experiment_config.yaml",
        help="YAML file with experiment configuration.",
    )

    args = parser.parse_args()

    with open(args.exp_file, "r") as f:
        cfg = yaml.safe_load(f)

    config = extract_experiment_configuration(cfg)

    timestamp_distribution = pd.read_csv(get_timestamp_behavior_path(config))
    userToExpDistribution = {
        user: expon(scale=row["median_timestamp_diff"])
        for user, row in timestamp_distribution.iterrows()
    }
    oracle_matrix = load_pickle_artifact(get_oracle_matrix_path(config))
    bootstrapped_df = load_bootstrapped_clicks(config)

    model = instantiate_model(config, hyperparameter_tuning_df=bootstrapped_df)

    sim = Simulator(
        oracle_matrix=oracle_matrix,
        model=model,
        initial_date=0.0,
        user_timestamp_distribution=userToExpDistribution,
        bootstrapped_df=bootstrapped_df,
        config=config,
    )
    simulated_df, maces, divergences, maps, coverages, mrrs, ginis, diversities = (
        sim.simulate(k=20)
    )

    print(f"Done! Saving simulated interactions...")
    base_artifacts_path = get_experiment_artifacts_path(config)
    if base_artifacts_path is not None and not os.path.exists(base_artifacts_path):
        os.makedirs(base_artifacts_path)
    simulated_df.to_pickle(base_artifacts_path / "simulated_interactions.pkl")

    save_pickle_artifact(maces, f"{base_artifacts_path}/maces.pkl")
    save_pickle_artifact(mrrs, f"{base_artifacts_path}/mrrs.pkl")
    save_pickle_artifact(divergences, f"{base_artifacts_path}/divergences.pkl")
    save_pickle_artifact(maps, f"{base_artifacts_path}/mace_at_k.pkl")
    save_pickle_artifact(coverages, f"{base_artifacts_path}/catalog_coverage.pkl")
    save_pickle_artifact(ginis, f"{base_artifacts_path}/gini.pkl")
    save_pickle_artifact(diversities, f"{base_artifacts_path}/diversities.pkl")
