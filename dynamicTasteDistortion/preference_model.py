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
from tqdm import tqdm
import torch


from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    MOVIELENS_PATH,
    STEAM_PATH,
    YELP_PATH,
    input_size_to_file_name,
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
)

from ioUtils import (
    get_or_create_oracle_matrix,
    get_or_create_oracle_model_artifacts,
    get_or_create_time_diff_df,
)
from dynamicTasteDistortion.scripts.data_utils import standardize_ids

DATA_TO_PATH = {"ml": MOVIELENS_PATH, "yelp": YELP_PATH, "steam": STEAM_PATH}

MODEL_NAME_TO_CLASS_NAME = {"NMF": NMF, "SVD++": SVDpp, "knn": KNNBasic}


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


def fit_evaluate(model, full_df, test_size=0.3):

    df_main_cols = full_df[["user", "item", "rating"]]
    trainset, testset = train_test_split(df_main_cols, test_size=test_size)
    reader = Reader(rating_scale=(1, 5))
    trainset = SurpriseDataset.load_from_df(trainset, reader).build_full_trainset()
    testset = list(testset.itertuples(index=False, name=None))

    fit_model = model.fit(trainset)
    predictions = model.test(testset)
    y_pred = [1 if pred.est >= 4 else 0 for pred in predictions]
    y_true = [1 if pred.r_ui >= 4 else 0 for pred in predictions]
    test_set_f1_score = f1_score(y_true, y_pred)

    return fit_model, test_set_f1_score


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets.")
    parser.add_argument(
        "--size",
        choices=["s", "m", "l"],
        required=True,
        help="Dataset size: s (1m), m (10m), l (20m).",
    )
    parser.add_argument(
        "--num_users",
        required=True,
        help="Number of users used to bootstrap clicks.",
    )

    parser.add_argument(
        "--data",
        choices=["ml", "yelp", "steam"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; steam",
    )
    args = parser.parse_args()
    data_type = args.data
    num_users = int(args.num_users)

    file_base_path = DATA_TO_PATH[data_type]
    file_size = input_size_to_file_name[args.size]
    file_path = f"{file_base_path}/{data_type}_{file_size}.pkl"
    print(f"Loading base dataset from {file_path}...")
    base_file = pd.read_pickle(file_path)
    print("Done!")
    candidates = base_file[USER_COL].unique().tolist()
    if num_users is not None:
        idx = torch.randperm(len(candidates))[:num_users]
        users = [candidates[i] for i in idx.tolist()]

    else:
        users = candidates

    base_df = base_file[base_file[USER_COL].isin(users)]
    df, user_id_map, _ = standardize_ids(base_df)
    users = [user_id_map[user] for user in users]

    print("Creating oracle model...")
    oracle_model = get_or_create_oracle_model_artifacts(df, data_type, file_size)

    print("Fitting and evaluating oracle model...")
    trained_model, f1_score_test = fit_evaluate(oracle_model, full_df=df, test_size=0.3)
    print(
        f"Model selection finished! model achieved f1 score of {f1_score_test:.2f} on test_set"
    )
    print("Creating filled oracle preference matrix...")
    _ = get_or_create_oracle_matrix(
        oracle_model=trained_model,
        df=df,
        data_type=data_type,
        file_size=file_size,
        users=users,
    )
    print("Defining user timestamp behaviour from source file...")
    get_or_create_time_diff_df(df, data_type, file_size, users)

    print("All done!")


if __name__ == "__main__":
    main()
