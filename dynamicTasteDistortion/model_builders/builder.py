from model_builders import MODEL_BUILDERS


def instantiate_model(config, hyperparameter_tuning_df):
    model_type = config["model_type"]
    try:
        builder = MODEL_BUILDERS[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {sorted(MODEL_BUILDERS)}"
        )
    return builder(config, hyperparameter_tuning_df)
