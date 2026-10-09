from pathlib import Path

from dynamicTasteDistortion.dataset_loader import input_size_to_sample_size
from dynamicTasteDistortion.ioUtils import (
    get_model_and_params_paths,
    load_pickle_artifact,
    get_cv_results_path,
    save_pickle_artifact,
)
from dynamicTasteDistortion.model_builders import MODEL_BUILDERS, PARAM_GRIDS
from dynamicTasteDistortion.scripts.tuner import HyperParameterTuner


def _load_existing_or_clear(config):
    """If saved artifacts exist, ask whether to reuse them.

    Returns the loaded model if the user keeps it, otherwise deletes the old
    artifacts and returns None (meaning: go tune again).
    """
    model_path, best_params_path = get_model_and_params_paths(config)

    if not Path(model_path).exists():
        return None

    file_size = input_size_to_sample_size[config["size"]]
    print(
        f"Best model {config['model_type']} found for {config['data_type']}_{file_size} "
        f"with {config['num_users']} users at {model_path}."
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
    return None


def instantiate_model(config, hyperparameter_tuning_df, k=5):
    model_type = config["model_type"]
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_type}'. Supported: {sorted(MODEL_BUILDERS)}"
        )

    # Fixed params given, or nothing to tune: build directly.
    if config.get("model_params") is not None or not PARAM_GRIDS.get(model_type):
        return MODEL_BUILDERS[model_type](config, hyperparameter_tuning_df)

    # Reuse a saved tuned model, unless the user chooses to retrain.
    existing = _load_existing_or_clear(config)
    if existing is not None:
        return existing

    print(f"Starting hyperparameter tuning for {model_type}.")
    tuner = HyperParameterTuner(hyperparameter_tuning_df, model_type, config=config)
    model, cv_results, best_params = tuner.tune(truth_set=hyperparameter_tuning_df, k=k)
    print(f"Tuned {model_type}: {best_params}")

    model_path, best_params_path = get_model_and_params_paths(config)
    save_pickle_artifact(best_params, best_params_path)
    save_pickle_artifact(model, model_path)
    cv_results.to_csv(get_cv_results_path(config))

    return model  # unfitted
