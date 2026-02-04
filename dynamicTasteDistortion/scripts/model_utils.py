import ast
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from surprise import NMF, Reader, SVDpp, Dataset as SurpriseDataset
from itertools import product

from tqdm import tqdm


from dynamicTasteDistortion.simulationConstants import (
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
)

MODEL_NAME_TO_CLASS_NAME = {"NMF": NMF, "SVD++": SVDpp}


class ModelChooser:
    def __init__(self, name):
        self.name = name
        self.models = {
            "SVD++": SVDpp,
            "NMF": NMF,
        }

        self.param_grid_svd = {
            "n_epochs": [10, 20],
            "lr_all": [0.002, 0.005],
            "reg_all": [0.02, 0.1],
        }

        self.param_grid_nmf = {
            "n_factors": [15, 30],
            "n_epochs": [50, 100],
            "reg_pu": [0.06, 0.1],
            "reg_qi": [0.06, 0.1],
        }

        self.model_name_to_params = {
            "SVD++": self.param_grid_svd,
            "NMF": self.param_grid_nmf,
        }

        self.model = self.models[self.name]
        self.params = self.model_name_to_params[self.name]

    def yield_models(self):
        model = self.model
        params = self.params
        param_names = list(params.keys())
        combinations = list(product(*params.values()))
        dicts = [dict(zip(param_names, values)) for values in combinations]
        return [(model(**param), param) for param in dicts]


def choose_best_model(df, data_type):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_results = {}

    model_names = ["SVD++", "NMF"]
    reader = Reader(rating_scale=(1, 5))

    usuarios = df["user"].unique()

    for model_name in model_names:
        model_config = ModelChooser(model_name)
        models = model_config.yield_models()

        for model, params in tqdm(
            models, desc=f"Starting optimization for model {model_name}..."
        ):
            f1_scores = []

            for train_users_idx, test_users_idx in kf.split(usuarios):
                train_users = set(usuarios[train_users_idx])
                test_users = set(usuarios[test_users_idx])

                trainset = df[df.user.isin(train_users)]
                testset = df[df.user.isin(test_users)]

                train_surprise = SurpriseDataset.load_from_df(
                    trainset[["user", "item", "rating"]], reader
                )
                trainset_surprise = train_surprise.build_full_trainset()

                testset_surprise = list(
                    testset[["user", "item", "rating"]].itertuples(
                        index=False, name=None
                    )
                )

                model.fit(trainset_surprise)
                predictions = model.test(testset_surprise)

                y_pred = [1 if pred.est >= 4 else 0 for pred in predictions]
                y_true = [1 if pred.r_ui >= 4 else 0 for pred in predictions]

                f1_scores.append(f1_score(y_true, y_pred))

            f1_results[(model_config, str(params))] = np.mean(f1_scores)

    f1_df = pd.DataFrame(
        [
            {"model": str(model_config.name), "params": params, "f1_score": score}
            for (model_config, params), score in f1_results.items()
        ]
    )
    destination_dir = f"{RESULTS_PATH}/{data_type}"
    model_artifacts_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}"
    # Mover isso pra main?
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not os.path.exists(model_artifacts_path):
        os.makedirs(model_artifacts_path)

    f1_df.to_pickle(f"{destination_dir}/oracle_model_f1_results.pkl")

    best_results = f1_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model = best_results.model
    best_params = ast.literal_eval(best_results.params)

    model_class = MODEL_NAME_TO_CLASS_NAME[best_model]
    oracle_model = model_class(**best_params)
    artifact = {"params": best_params, "model_class": model_class}
    with open(f"{MODEL_ARTIFACTS_PATH}/{data_type}/oracle_model_params.pkl", "wb") as f:
        pickle.dump(artifact, f)

    return oracle_model
