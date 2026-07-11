import torch
import re
from pathlib import Path

from dynamicTasteDistortion.scripts.metrics_utils import remove_outliers
from dynamicTasteDistortion.scripts.model_utils import (
    choose_best_model,
)
from dynamicTasteDistortion.dataset_loader import load_df, input_size_to_sample_size


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

from dynamicTasteDistortion.scripts.model_utils import (
    HyperParameterTuner,
    MostPopularRecommender,
)

from bprMf.bpr_mf import bprMFWithClickDebiasing, bprMf


from dynamicTasteDistortion.scripts.data_utils import standardize_ids

import os
from pathlib import Path

import pandas as pd
import pickle

import yaml


def get_model_and_params_paths(config):
    """
    Extract model and params paths from config.

    Args:
        config: Configuration dictionary containing data_type, file_size,
                num_users, and model_type keys.

    Returns:
        tuple: (model_path, best_params_path) as strs.
    """
    model_path = get_model_path(config)
    best_params_path = get_best_params_path(config)

    return model_path, best_params_path


def read_experiment(exp_file):
    with open(exp_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_experiment_configuration(cfg):
    model_type = cfg.get("model", "bpr")
    data_type = cfg["data_type"]
    size = cfg["size"]
    file_size = input_size_to_file_name[size]
    n_examination_trials = int(cfg.get("examination_attempts", 3))

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
    n_trials = int(cfg.get("n_trials", 1))
    return {
        "model_type": model_type,
        "data_type": data_type,
        "size": size,
        "file_size": file_size,
        "n_examination_trials": n_examination_trials,
        "exp_name": exp_name,
        "calibration_type": calibration_type,
        "rounds": rounds,
        "num_rounds_per_eval": num_rounds_per_eval,
        "num_users": num_users,
        "model_params": model_params,
        "overwrite_model_selection": overwrite_model_selection,
        "preference_update_rate": preference_update_rate,
        "n_trials": n_trials,
    }


def read_metrics(cfg_file, remove_outliers=False):
    # TODO: reimplementar o remove_outliers
    base_artifacts_path = get_experiment_artifacts_path(cfg_file)
    return load_pickle_artifact(f"{base_artifacts_path}/all_results.pkl")


def read_metrics_per_seed(cfg_file, seed=42):
    base_artifacts_path = get_experiment_artifacts_path(cfg_file)
    # simulated_df.to_pickle(
    #     base_artifacts_path / f"simulated_interactions_seed{seed}.pkl"
    # )
    maces = load_pickle_artifact(f"{base_artifacts_path}/maces_seed{seed}.pkl")

    divergences = load_pickle_artifact(
        f"{base_artifacts_path}/divergences_seed{seed}.pkl"
    )

    coverages = load_pickle_artifact(
        f"{base_artifacts_path}/catalog_coverage_seed{seed}.pkl"
    )
    ginis = load_pickle_artifact(f"{base_artifacts_path}/gini_seed{seed}.pkl")

    diversities = load_pickle_artifact(
        f"{base_artifacts_path}/diversities_seed{seed}.pkl"
    )

    frags = load_pickle_artifact(f"{base_artifacts_path}/frags_seed{seed}.pkl")

    return {
        "maces": maces,
        "divergences": divergences,
        "coverages": coverages,
        "ginis": ginis,
        "diversities": diversities,
        "fragmentation": frags,
    }


def read_simulated_interactions_per_experiment_seed(cfg_file, seed=42):
    base_artifacts_path = get_experiment_artifacts_path(cfg_file)
    simulated_df = load_pickle_artifact(
        base_artifacts_path / f"simulated_interactions_seed{seed}.pkl"
    )
    return simulated_df


def load_bootstrapped_clicks(cfg):
    data_type = cfg["data_type"]
    file_size = cfg["file_size"]
    num_users = cfg["num_users"]
    output_path = f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_bootstrapped.pkl"
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


def get_base_model_path(cfg):
    data_type = cfg["data_type"]
    file_size = cfg["file_size"]
    num_users = cfg["num_users"]
    model_type = cfg["model_type"]
    return f"{MODEL_ARTIFACTS_PATH}/{model_type}_{data_type}_{file_size}_n_users={num_users}"


def get_model_path(cfg):
    base_path = get_base_model_path(cfg)
    return f"{base_path}_model.pkl"


def get_best_params_path(cfg):
    base_path = get_base_model_path(cfg)
    return f"{base_path}_params.pkl"


def get_cv_results_path(cfg):
    base_path = get_base_model_path(cfg)
    return f"{base_path}_cv_results.csv"


def get_oracle_matrix_path(cfg=None, data_type=None, file_size=None, num_users=None):
    if cfg is not None:
        data_type = cfg["data_type"]
        file_size = cfg["file_size"]
        num_users = cfg["num_users"]
    elif None in (data_type, file_size, num_users):
        raise ValueError(
            "Either cfg or all of data_type, file_size, num_users must be provided"
        )
    return f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_oracle.pkl"


def get_ids_mapping_path(data_type, file_size, num_users):
    return f"{SIMULATION_PATH}/{data_type}_{file_size}_n_users={num_users}_mapping"


def get_user_id_to_idx_mapping_path(data_type, file_size, num_users):
    path = get_ids_mapping_path(data_type, file_size, num_users)
    return f"{path}_user_id_to_idx.pkl"


def get_item_id_to_idx_mapping_path(data_type, file_size, num_users):
    path = get_ids_mapping_path(data_type, file_size, num_users)
    return f"{path}_item_id_to_idx.pkl"


def get_user_idx_to_id_mapping_path(data_type, file_size, num_users):
    path = get_ids_mapping_path(data_type, file_size, num_users)
    return f"{path}_user_idx_to_id.pkl"


def get_item_idx_to_id_mapping_path(data_type, file_size, num_users):
    path = get_ids_mapping_path(data_type, file_size, num_users)
    return f"{path}_item_idx_to_id.pkl"


def get_timestamp_behavior_path(
    cfg=None, data_type=None, file_size=None, num_users=None
):
    if cfg is not None:
        data_type = cfg["data_type"]
        file_size = cfg["file_size"]
        num_users = cfg["num_users"]
    elif None in (data_type, file_size, num_users):
        raise ValueError(
            "Either cfg or all of data_type, file_size, num_users must be provided"
        )
    return f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}_n_users={num_users}_avg_time_diff.csv"


def get_or_create_oracle_matrix(oracle_model, df, data_type, file_size, users):
    num_users = len(users)
    if data_type == "food":
        class_cutoff = 3.0
    elif data_type == "globo":
        # This is the only dataset that is based on implicit feedback.
        class_cutoff = 0.5
        rating_scale = (0, 1)
    else:
        class_cutoff = 4.0

    oracle_output_path = get_oracle_matrix_path(
        data_type=data_type, file_size=file_size, num_users=num_users
    )
    if os.path.exists(oracle_output_path):
        print(
            f"Filled oracle matrix for {data_type}_{file_size} that uses {num_users} users already exists! Skipping matrix filling."
        )
        filled_oracle_matrix = load_pickle_artifact(oracle_output_path)
    else:
        user_id_to_idx_map = load_pickle_artifact(
            get_user_id_to_idx_mapping_path(
                data_type=data_type, file_size=file_size, num_users=num_users
            )
        )
        item_id_to_idx_map = load_pickle_artifact(
            get_item_id_to_idx_mapping_path(
                data_type=data_type, file_size=file_size, num_users=num_users
            )
        )
        print("Filling up rating matrix...")
        filled_oracle_matrix = fill_out_matrix(
            base_df=df,
            model=oracle_model,
            user_sample=users,
            rating_cutoff=class_cutoff,
            rating_scale=rating_scale,
            item_id_to_idx_map=item_id_to_idx_map,
            user_id_to_idx_map=user_id_to_idx_map,
        )
        print(f"Writing filled out matrix to {oracle_output_path}")
    filled_oracle_matrix.to_pickle(oracle_output_path)
    return filled_oracle_matrix


def instantiate_model(config, hyperparameter_tuning_df):

    model_type_to_class = {"bpr": bprMFWithClickDebiasing, "bpr_classic": bprMf}
    overwrite_model_selection = config["overwrite_model_selection"]
    model_params = config["model_params"]
    model_type = config["model_type"]
    data_type = config["data_type"]
    overwrite_model_selection = config["overwrite_model_selection"]
    size = config["size"]
    num_users = config["num_users"]
    file_size = input_size_to_sample_size[size]

    n_users = hyperparameter_tuning_df.user.max() + 1
    n_items = hyperparameter_tuning_df.item.max() + 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model_path, best_params_path = get_model_and_params_paths(config)

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
        sample_size = size if data_type == "ml" else file_size
        df = load_df(data_type, size=sample_size)
        processed_df, _, _ = standardize_ids(df)
        model = MostPopularRecommender(processed_df)

    elif model_type in ("bpr", "bpr_classic"):
        ModelClass = model_type_to_class[model_type]

        if Path(model_path).exists():
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
                Path(model_path).unlink()
                Path(best_params_path).unlink()
                cv_results_save_path = get_cv_results_path(config)
                cv_results_path_obj = Path(cv_results_save_path)
                if cv_results_path_obj.exists():
                    cv_results_path_obj.unlink()

        if not Path(model_path).exists():  # either never existed, or just deleted above
            print(
                f"No {model_type} found for {data_type}_{file_size} with {num_users} sampled users. Starting hyperparameter tuning."
            )
            tuner = HyperParameterTuner(hyperparameter_tuning_df, ModelClass)
            model, cv_results, best_params = tuner.tune(
                truth_set=hyperparameter_tuning_df, k=5
            )
            save_pickle_artifact(best_params, best_params_path)
            save_pickle_artifact(model, model_path)
            cv_results_save_path = get_cv_results_path(config)
            cv_results.to_csv(cv_results_save_path)
            model = ModelClass(
                num_users=n_users, num_items=n_items, dev=dev, **best_params
            )

    else:
        model = None

    return model


def get_missing_seeds(base_artifacts_path, predetermined_seeds):
    if not base_artifacts_path.exists():
        return list(predetermined_seeds)

    seed_pattern = re.compile(r"_seed(\d+)\.pkl$")

    found_seeds = set()
    for file in base_artifacts_path.iterdir():
        if not file.is_file():
            continue
        match = seed_pattern.search(file.name)
        if match:
            found_seeds.add(int(match.group(1)))

    missing_seeds = [seed for seed in predetermined_seeds if seed not in found_seeds]
    return missing_seeds


def get_experiment_artifacts_path(config):
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

    return base_artifacts_path


def get_or_create_oracle_model_artifacts(df, data_type, size):
    oracle_model_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/oracle_model.pkl"

    if os.path.exists(oracle_model_path):
        print(
            f"Oracle model trained on {data_type}_{size} found!Skipping model selection"
        )
        oracle_model = load_pickle_artifact(oracle_model_path)
    else:
        os.makedirs(f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}/", exist_ok=True)
        print("Starting model selection...")
        rating_scale = (1, 5)
        if data_type == "food":
            class_cutoff = 3.0
        elif data_type == "globo":
            # This is the only dataset that is based on implicit feedback.
            class_cutoff = 0.5
            rating_scale = (0, 1)
            # We also convert genres into list of strings for
        else:
            class_cutoff = 4.0
        oracle_model, f1_results = choose_best_model(
            df, class_cutoff=class_cutoff, rating_scale=rating_scale
        )
        save_pickle_artifact(oracle_model, oracle_model_path)
        destination_dir = f"{RESULTS_PATH}/{data_type}_{size}"
        os.makedirs(destination_dir, exist_ok=True)
        f1_results.to_pickle(f"{destination_dir}/oracle_model_f1_results.pkl")
        print(
            f"Saving oracle model f1 results to {destination_dir}/oracle_model_f1_results.pkl"
        )
    return oracle_model


def get_or_create_time_diff_df(df, data_type, size, num_users):
    timestamp_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}_{size}_n_users={num_users}_avg_time_diff.csv"
    if os.path.exists(timestamp_path):
        print(
            f"Timestamp behavior for {data_type}_{size} that uses {num_users} users already exists! Skipping timestamp behavior calculation."
        )
        avg_std_time_diff_per_user = pd.read_csv(timestamp_path)
    else:
        avg_std_time_diff_per_user = get_timestamp_behavior(df=df)

        if len(avg_std_time_diff_per_user) == 0:
            print("Timestamp df is emtpy! Please check get_timestamp_behavior func")
            return
        print(f"timestamp diff: {avg_std_time_diff_per_user}")
        print(f"Writing timestamp behavior per user to {timestamp_path}")
        avg_std_time_diff_per_user.to_csv(
            timestamp_path,
            index=False,
        )
