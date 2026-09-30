from dynamicTasteDistortion.model_builders import MODEL_BUILDERS, PARAM_GRIDS
from dynamicTasteDistortion.scripts.model_utils import HyperParameterTuner


def instantiate_model(config, hyperparameter_tuning_df, k=5):
    model_type = config["model"]
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_type}'. Supported: {sorted(MODEL_BUILDERS)}"
        )

    # Fixed params given, or nothing to tune: build directly.
    if config.get("model_params") is not None or not PARAM_GRIDS.get(model_type):
        return MODEL_BUILDERS[model_type](config, hyperparameter_tuning_df)

    tuner = HyperParameterTuner(hyperparameter_tuning_df, model_type, config=config)
    model, _, best_params = tuner.tune(truth_set=hyperparameter_tuning_df, k=k)
    print(f"Tuned {model_type}: {best_params}")
    return model  # unfitted
