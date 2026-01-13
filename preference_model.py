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
    USER_COL,
    MOVIELENS_PATH,
    STEAM_PATH,
    YELP_PATH,
    input_size_to_file_name,
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
    SIMULATION_PATH,
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


def fill_out_matrix(df, model):
    reader = Reader(rating_scale=(1, 5))
    trainset = SurpriseDataset.load_from_df(
        df[["user", "item", "rating"]], reader
    ).build_full_trainset()
    model.fit(trainset)
    all_users = trainset.all_users()
    all_items = trainset.all_items()

    user_ids = [trainset.to_raw_uid(u) for u in all_users]
    item_ids = [trainset.to_raw_iid(i) for i in all_items]

    predictions = []
    for user_id in tqdm(user_ids, desc="Predicting missing ratings"):
        surprise_internal_user_id = trainset.to_inner_uid(user_id)
        rated_item_ids = set(
            [
                trainset.to_raw_iid(item)
                for item, _ in trainset.ur[surprise_internal_user_id]
            ]
        )
        for item_id in item_ids:
            if item_id not in rated_item_ids:
                # predict the rating
                pred = model.predict(user_id, item_id)
                predictions.append([user_id, item_id, pred])

    # binarize predictions
    processed_predictions = [
        [pred[0], pred[1], int(pred[2].est >= 4)] for pred in predictions
    ]

    predictions_df = pd.DataFrame(
        processed_predictions, columns=["user", "item", "rating"]
    )

    # Aqui vai dar problema. Genres não é hashable por ser lista.
    genres_df = df[["item", "genres"]].drop_duplicates(subset="item")
    predictions_df = predictions_df.merge(genres_df, on="item")
    base_df = df[["user", "item", "genres", "rating"]]
    base_df.loc[:, "rating"] = base_df["rating"].apply(lambda rating: int(rating >= 4))

    # Combine missing entries with previously filled
    df_filled = pd.concat([base_df, predictions_df], ignore_index=True)

    return df_filled


def get_timestamp_behavior(df):
    avg_std_time_diff_per_user = (
        df.sort_values([USER_COL, "timestamp"])
        .groupby(USER_COL)["timestamp"]
        .agg(
            median_timestamp_diff=lambda x: (
                np.median(np.diff(x)) if len(x) > 1 else np.nan
            ),
            std_timestamp_diff=lambda x: np.diff(x).std() if len(x) > 1 else np.nan,
            n_entries="count",
        )
        .reset_index()
    )

    positive_timestamp_diff = list(
        avg_std_time_diff_per_user[
            avg_std_time_diff_per_user["median_timestamp_diff"] > 0
        ][USER_COL]
    )

    global_median_timestamp_diff = np.median(
        np.diff(
            df[df[USER_COL].isin(positive_timestamp_diff)].sort_values(
                [USER_COL, "timestamp"]
            )["timestamp"]
        )
    )

    avg_std_time_diff_per_user["median_timestamp_diff"] = avg_std_time_diff_per_user[
        "median_timestamp_diff"
    ].replace(0, global_median_timestamp_diff)

    return avg_std_time_diff_per_user


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
    print("Loading base dataset...")
    file_base_path = DATA_TO_PATH[data_type]
    file_size = input_size_to_file_name[args.size]
    file_path = f"{file_base_path}/{data_type}_{file_size}.pkl"
    base_file = pd.read_pickle(file_path)
    print("Done!")
    params_path = (
        f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}/oracle_model_params.pkl"
    )
    if os.path.exists(params_path):
        print(
            f"Oracle model trained on {data_type}_{file_size} found!Skipping model selection"
        )
        with open(params_path, "rb") as f:
            oracle_model_artifact = pickle.load(f)

        model_class = oracle_model_artifact["model_class"]
        model_params = oracle_model_artifact["params"]
        oracle_model = model_class(**model_params)
    else:
        print("Starting model selection...")
        oracle_model = choose_best_model(base_file, f"{data_type}_{file_size}")

    print("Fitting and evaluating oracle model...")
    trained_model, f1_score_test = fit_evaluate(
        oracle_model, full_df=base_file, test_size=0.3
    )
    print(
        f"Model selection finished! model achieved f1 score of {f1_score_test:.2f} on test_set"
    )
    output_path = f"{SIMULATION_PATH}/{data_type}_{file_size}_oracle.pkl"
    if os.path.exists(output_path):
        print(
            f"Filled oracle matrix for {data_type}_{file_size} already exists! Skipping matrix filling."
        )
        with open(output_path, "rb") as f:
            filled_oracle_matrix = pickle.load(f)
    else:
        print("Filling up rating matrix...")
        filled_oracle_matrix = fill_out_matrix(df=base_file, model=trained_model)
        print(f"Writing filled out matrix to {output_path}")
        filled_oracle_matrix.to_pickle(output_path)
    print("Defining user timestamp behaviour from source file...")
    timestamp_output_path = (
        f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}/avg_time_diff.csv"
    )
    if os.path.exists(timestamp_output_path):
        print(
            f"Timestamp behavior for {data_type}_{file_size} already exists! Skipping timestamp behavior calculation."
        )
        avg_std_time_diff_per_user = pd.read_csv(timestamp_output_path)
    else:
        avg_std_time_diff_per_user = get_timestamp_behavior(df=base_file)
        print(f"Writing timestamp behavior per user to {timestamp_output_path}")
        avg_std_time_diff_per_user.to_csv(
            f"{MODEL_ARTIFACTS_PATH}/{data_type}_{file_size}/avg_time_diff.csv",
            index=False,
        )

    print("All done!")


if __name__ == "__main__":
    main()
