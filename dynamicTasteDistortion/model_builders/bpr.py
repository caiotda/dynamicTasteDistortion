import torch
from pathlib import Path

DEFAULT_PARAMS = {
    "factors": 64,
    "lr": 0.0001,
    "reg_lambda": 0.001,
    "num_negatives": 5,
    "n_epochs": 7,
}


def build_bpr(config, hyperparameter_tuning_df, model_class, **params):
    """Construct an unfitted BPR model.

    Param precedence (later wins): DEFAULT_PARAMS < config["model_params"] < params
    (the tuner passes candidate hyperparameters through **params).
    """
    n_users = hyperparameter_tuning_df.user.max() + 1
    n_items = hyperparameter_tuning_df.item.max() + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model_params = {
        **DEFAULT_PARAMS,
        **(config.get("model_params") or {}),
        **params,
    }
    return model_class(num_users=n_users, num_items=n_items, dev=dev, **model_params)
