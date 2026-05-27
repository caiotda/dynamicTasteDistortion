import argparse
import pandas as pd
from pathlib import Path

from dynamicTasteDistortion.simulationConstants import (
    RESULTS_PATH,
)
from dynamicTasteDistortion.simulation.simulator import Simulator

from dynamicTasteDistortion.ioUtils import (
    get_oracle_matrix_path,
    get_timestamp_behavior_path,
    instantiate_model,
    load_bootstrapped_clicks,
    load_pickle_artifact,
    save_pickle_artifact,
    extract_experiment_configuration,
)

from dynamicTasteDistortion.dataset_loader import input_size_to_sample_size

from scipy.stats import expon


from dynamicTasteDistortion.scripts.data_utils import standardize_ids
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

    data_type = config["data_type"]
    size = config["size"]
    file_size = input_size_to_sample_size[size]
    exp_name = config["exp_name"]
    rounds = config["rounds"]
    num_rounds_per_eval = config["num_rounds_per_eval"]
    num_users = config["num_users"]
    
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
        config=config,
    )
    simulated_df, maces, divergences, maps, coverages, mrrs, ginis, diversities = (
        sim.simulate(L=num_rounds_per_eval, rounds=rounds, k=20)
    )

    print(f"Done! Saving simulated interactions...")
    simulated_df.to_pickle(base_artifacts_path / "simulated_interactions.pkl")

    save_pickle_artifact(maces, f"{base_artifacts_path}/maces.pkl")
    save_pickle_artifact(mrrs, f"{base_artifacts_path}/mrrs.pkl")
    save_pickle_artifact(divergences, f"{base_artifacts_path}/divergences.pkl")
    save_pickle_artifact(maps, f"{base_artifacts_path}/mace_at_k.pkl")
    save_pickle_artifact(coverages, f"{base_artifacts_path}/catalog_coverage.pkl")
    save_pickle_artifact(ginis, f"{base_artifacts_path}/gini.pkl")
    save_pickle_artifact(diversities, f"{base_artifacts_path}/diversities.pkl")
