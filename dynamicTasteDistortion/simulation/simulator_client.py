import argparse
import pandas as pd
import torch
import pickle
from pathlib import Path

from dynamicTasteDistortion.simulationConstants import (
    input_size_to_file_name,
    RESULTS_PATH,
)
from dynamicTasteDistortion.simulation.simulator import Simulator

from dynamicTasteDistortion.ioUtils import (
    get_best_params_path,
    get_model_path,
    get_oracle_matrix_path,
    get_timestamp_behavior_path,
    load_bootstrapped_clicks,
    load_pickle_artifact,
    get_cv_results_path,
    save_pickle_artifact,
)

from dynamicTasteDistortion.dataset_loader import load_df, input_size_to_sample_size


from scipy.stats import expon

from bprMf.bpr_mf import bprMFWithClickDebiasing, bprMf
from dynamicTasteDistortion.scripts.model_utils import (
    HyperParameterTuner,
    MostPopularRecommender,
)
from dynamicTasteDistortion.scripts.data_utils import standardize_ids
import yaml

model_type_to_class = {"bpr": bprMFWithClickDebiasing, "bpr_classic": bprMf}


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

    rounds = int(cfg["rounds"])
    num_rounds_per_eval = int(cfg["num_rounds_per_eval"])
    num_users = int(cfg["num_users"])
    model_params = cfg.get("params", None)
    overwrite_model_selection = True if model_params is not None else False

    preference_update_rate = float(cfg.get("preference_update_rate", 0))
    compare_to_h_0 = True if cfg.get("compare_to_h_0", "y") == "y" else False

    timestamp_distribution = pd.read_csv(
        get_timestamp_behavior_path(data_type, file_size, num_users)
    )
    oracle_matrix = load_pickle_artifact(
        get_oracle_matrix_path(data_type, file_size, num_users)
    )
    bootstrapped_df = load_bootstrapped_clicks(data_type, file_size, num_users)

    model_path = get_model_path(data_type, file_size, num_users, model_type)
    best_params_path = get_best_params_path(data_type, file_size, num_users, model_type)
    model_path_obj = Path(model_path)
    best_params_path_obj = Path(best_params_path)

    n_users = bootstrapped_df.user.max() + 1
    n_items = bootstrapped_df.item.max() + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if overwrite_model_selection:
        if model_type not in ("bpr", "bpr_classic"):
            raise ValueError(
                "overwrite_model_selection is not supported for most_popular or random model."
            )
        ModelClass = model_type_to_class[model_type]
        print(f"Using predefined best params: {model_params}")
        model = ModelClass(
            num_users=n_users, num_items=n_items, dev=dev, **model_params
        )

    elif model_type == "most_popular":
        print(f"Loading {data_type}_{file_size} dataset to fit Most Popular model...")
        sample_size = size if data_type == "ml" else input_size_to_sample_size[size]
        df = load_df(data_type, size=sample_size)
        processed_df, _, _ = standardize_ids(df)
        model = MostPopularRecommender(processed_df)

    elif model_type in ("bpr", "bpr_classic"):
        ModelClass = model_type_to_class[model_type]

        if model_path_obj.exists():
            print(
                f"Best model {model_type} found for {data_type}_{file_size} with {num_users} at {model_path}."
            )
            print(f"Model params: {load_pickle_artifact(best_params_path)}")
            overwrite = (
                input("Model artifact already exists. Overwrite and retrain? [y/N]: ")
                .strip()
                .lower()
            )

            if overwrite in ("n", "no", ""):
                print("Using existing model and skipping hyperparameter tuning.")
                model = load_pickle_artifact(model_path)
            else:
                print("Deleting existing model artifact and retraining.")
                model_path_obj.unlink()
                best_params_path_obj.unlink()
                cv_results_save_path = get_cv_results_path(
                    data_type, file_size, num_users, model_type
                )
                cv_results_path_obj = Path(cv_results_save_path)
                if cv_results_path_obj.exists():
                    cv_results_path_obj.unlink()

        if not model_path_obj.exists():  # either never existed, or just deleted above
            print(
                f"No {model_type} found for {data_type}_{file_size} with {num_users} sampled users. Starting hyperparameter tuning."
            )
            tuner = HyperParameterTuner(bootstrapped_df, ModelClass)
            model, cv_results, best_params = tuner.tune(truth_set=bootstrapped_df, k=5)
            save_pickle_artifact(best_params, best_params_path)
            save_pickle_artifact(model, model_path)
            cv_results_save_path = get_cv_results_path(
                data_type, file_size, num_users, model_type
            )
            cv_results.to_csv(cv_results_save_path)
            model = ModelClass(
                num_users=n_users, num_items=n_items, dev=dev, **best_params
            )

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
        calibration_type=calibration_type,
        preference_update_rate=preference_update_rate,
        compare_to_h_0=compare_to_h_0,
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
