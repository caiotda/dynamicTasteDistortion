import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from surprise import KNNBasic, NMF, Reader, SVDpp, Dataset as SurpriseDataset
import torch


from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    MOVIELENS_PATH,
    FOOD_PATH,
    YELP_PATH,
    input_size_to_file_name,
)

from dynamicTasteDistortion.ioUtils import (
    get_or_create_oracle_matrix,
    get_or_create_oracle_model_artifacts,
    get_or_create_time_diff_df,
)
from dynamicTasteDistortion.scripts.data_utils import standardize_ids

DATA_TO_PATH = {"ml": MOVIELENS_PATH, "yelp": YELP_PATH, "food": FOOD_PATH}


def split_train_test_per_user(df, train_frac=0.8):
    train_parts = []
    test_parts = []

    for _, user_df in df.groupby("user"):
        user_df = user_df.sort_values("timestamp")

        n_train = int(len(user_df) * train_frac)

        train_parts.append(user_df.iloc[:n_train])
        test_parts.append(user_df.iloc[n_train:])

    train_df = pd.concat(train_parts).reset_index(drop=True)

    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def fit_evaluate(model, full_df, test_size=0.3, class_cutoff=4.0):

    df_main_cols = full_df[["user", "item", "rating"]]
    trainset, testset = train_test_split(df_main_cols, test_size=test_size)
    reader = Reader(rating_scale=(1, 5))
    trainset = SurpriseDataset.load_from_df(trainset, reader).build_full_trainset()
    testset = list(testset.itertuples(index=False, name=None))

    fit_model = model.fit(trainset)
    predictions = model.test(testset)
    y_pred = [1 if pred.est >= class_cutoff else 0 for pred in predictions]
    y_true = [1 if pred.r_ui >= class_cutoff else 0 for pred in predictions]
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
        choices=["ml", "yelp", "food"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; food",
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

    df, _, _ = standardize_ids(base_file)
    print("Creating oracle model...")
    oracle_model = get_or_create_oracle_model_artifacts(df, data_type, file_size)

    if data_type != "food":
        class_cutoff = 3.0
    else:
        class_cutoff = 4.0
    print("Fitting and evaluating oracle model...")
    trained_model, f1_score_test = fit_evaluate(
        oracle_model, full_df=df, test_size=0.3, class_cutoff=class_cutoff
    )
    print(
        f"Model selection finished! model achieved f1 score of {f1_score_test:.2f} on test_set"
    )
    print(
        f"Creating filled oracle preference matrix for sampel of {num_users} users..."
    )
    candidates = df[USER_COL].unique().tolist()

    if num_users is not None:
        idx = torch.randperm(len(candidates))[:num_users]
        users = [candidates[i] for i in idx.tolist()]

    else:
        users = candidates

    prediction_df = df[df[USER_COL].isin(users)]
    _ = get_or_create_oracle_matrix(
        oracle_model=trained_model,
        df=prediction_df,
        data_type=data_type,
        file_size=file_size,
        users=users,
    )
    print("Defining user timestamp behaviour from source file...")
    get_or_create_time_diff_df(df, data_type, file_size, users)

    print("All done!")


if __name__ == "__main__":
    main()
