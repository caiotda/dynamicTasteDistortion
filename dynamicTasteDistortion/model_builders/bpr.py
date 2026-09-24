import torch
from pathlib import Path

from dynamicTasteDistortion.dataset_loader import input_size_to_sample_size
from dynamicTasteDistortion.ioUtils import (
    get_model_and_params_paths,
    load_pickle_artifact,
    get_cv_results_path,
    save_pickle_artifact,
)


from dynamicTasteDistortion.scripts.model_utils import HyperParameterTuner


def build_bpr(config, hyperparameter_tuning_df, model_class):
    model_type = config["model_type"]
    data_type = config["data_type"]
    size = config["size"]
    num_users = config["num_users"]
    file_size = input_size_to_sample_size[size]

    n_users = hyperparameter_tuning_df.user.max() + 1
    n_items = hyperparameter_tuning_df.item.max() + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model_path, best_params_path = get_model_and_params_paths(config)

    if config["overwrite_model_selection"]:
        model_params = config["model_params"]
        print(f"Using predefined best params: {model_params}")
        return model_class(
            num_users=n_users, num_items=n_items, dev=dev, **model_params
        )

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
            return load_pickle_artifact(model_path)

        print("Deleting existing model artifact and retraining.")
        Path(model_path).unlink()
        Path(best_params_path).unlink()
        cv_results_path = Path(get_cv_results_path(config))
        if cv_results_path.exists():
            cv_results_path.unlink()

    # either never existed, or just deleted above
    print(
        f"No {model_type} found for {data_type}_{file_size} with {num_users} sampled users. Starting hyperparameter tuning."
    )
    tuner = HyperParameterTuner(hyperparameter_tuning_df, model_class)
    model, cv_results, best_params = tuner.tune(truth_set=hyperparameter_tuning_df, k=5)
    save_pickle_artifact(best_params, best_params_path)
    save_pickle_artifact(model, model_path)
    cv_results.to_csv(get_cv_results_path(config))

    return model_class(num_users=n_users, num_items=n_items, dev=dev, **best_params)
