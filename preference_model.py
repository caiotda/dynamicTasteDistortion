import argparse
import ast
import pickle
import os

import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from surprise import KNNBasic, NMF, Reader, SVDpp, Dataset as SurpriseDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from simulationConstants import (
    MOVIELENS_PATH,
    STEAM_PATH,
    YELP_PATH,
    input_size_to_file_name,
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
)

DATA_TO_PATH = {"ml": MOVIELENS_PATH, "yelp": YELP_PATH, "steam": STEAM_PATH}

MODEL_NAME_TO_CLASS_NAME = {"NMF": NMF, "SVD++": SVDpp, "knn": KNNBasic}


class ModelChooser:
    def __init__(self, name):
        self.name = name
        self.models = {
            "SVD++": SVDpp,
            "NMF": NMF,
            "knn": KNNBasic,
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

        sim_functions = ["cosine", "pearson"]
        user_based = [True, False]

        sim_options_combinations = [
            {"name": sim, "user_based": ub}
            for sim in sim_functions
            for ub in user_based
        ]

        self.param_grid_knn = {
            "k": [20, 40, 60],
            "sim_options": sim_options_combinations,
        }

        self.model_name_to_params = {
            "SVD++": self.param_grid_svd,
            "NMF": self.param_grid_nmf,
            "knn": self.param_grid_knn,
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


def split_train_test_per_user(df, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)

    users = df.user.unique()
    rng.shuffle(users)

    n_train = int(len(users) * train_frac)
    users_train = set(users[:n_train])
    users_test = set(users[n_train:])

    train_df = df[df.user.isin(users_train)].reset_index(drop=True)
    test_df = df[df.user.isin(users_test)].reset_index(drop=True)

    return train_df, test_df


def choose_best_model(df, data_type):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_results = {}

    model_names = ["SVD++", "NMF", "knn"]
    reader = Reader(rating_scale=(1, 5))

    usuarios = df["user"].unique()

    for model_name in model_names:
        model_config = ModelChooser(model_name)
        models = model_config.yield_models()

        for model, params in models:
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

    f1_df.to_pickle(f"{RESULTS_PATH}/{data_type}/oracle_model_f1_results.pkl")

    best_results = f1_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model = best_results.model
    best_params = ast.literal_eval(best_results.params)

    model_class = MODEL_NAME_TO_CLASS_NAME[best_model]
    oracle_model = model_class(**best_params)
    pickle.dump(oracle_model, f"{MODEL_ARTIFACTS_PATH}/{data_type}/oracle_model.pkl")
    return oracle_model


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets.")
    parser.add_argument(
        "--size",
        choices=["s", "m", "l"],
        required=True,
        help="Dataset size: s (1m), m (10m), l (20m).",
    )
    parser.add_argument(
        "--data",
        choices=["ml", "yelp", "steam"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; steam",
    )
    args = parser.parse_args()
    data_type = args.data
    file_base_path = DATA_TO_PATH[data_type]
    file_size = input_size_to_file_name[args.size]
    file_path = f"{file_base_path}/{data_type}_{file_size}.pkl"
    base_file = pd.read_pickle(file_path)
    model_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}/oracle_model.pkl"
    if os.path.exists(model_path):
        print(
            f"Oracle model trained on {data_type}_{file_size} found!Skipping model selection"
        )
        oracle_model = pickle.load(model_path)
    else:
        oracle_model = choose_best_model(base_file, f"{data_type}_{file_size}")


if __name__ == "__main__":
    main()
