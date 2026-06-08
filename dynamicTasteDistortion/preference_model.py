import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from surprise import KNNBasic, NMF, Reader, SVDpp, Dataset as SurpriseDataset
import torch


from dynamicTasteDistortion.simulationConstants import (
    ITEM_COL,
    USER_COL,
    MOVIELENS_PATH,
    FOOD_PATH,
    YELP_PATH,
    GLOBO_PATH,
    input_size_to_file_name,
)

from dynamicTasteDistortion.ioUtils import (
    get_or_create_oracle_matrix,
    get_or_create_oracle_model_artifacts,
    get_or_create_time_diff_df,
    get_user_id_to_idx_mapping_path,
    get_item_id_to_idx_mapping_path,
    get_user_idx_to_id_mapping_path,
    get_item_idx_to_id_mapping_path,
    save_pickle_artifact
)

DATA_TO_PATH = {
    "ml": MOVIELENS_PATH,
    "yelp": YELP_PATH,
    "food": FOOD_PATH,
    "globo": GLOBO_PATH,
}


def fit_evaluate(model, full_df, test_size=0.3, class_cutoff=4.0, rating_scale=(1, 5)):

    df_main_cols = full_df[["user", "item", "rating"]]
    trainset, testset = train_test_split(df_main_cols, test_size=test_size)
    reader = Reader(rating_scale=rating_scale)
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
        choices=["ml", "yelp", "food", "globo"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; food; globo.com",
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

    print("Creating oracle model...")
    # We create the oracle based on the entire dataset, in order to avoid removing important neighborhood information
    oracle_model = get_or_create_oracle_model_artifacts(df=base_file, data_type=data_type, size=file_size)
    rating_scale = (1, 5)
    if data_type == "food":
        class_cutoff = 3.0
    elif data_type == "globo":
        # This is the only dataset that is based on implicit feedback.
        rating_scale = (0, 1)
        class_cutoff = 0.5
    else:
        class_cutoff = 4.0
    print("Fitting and evaluating oracle model...")
    trained_model, f1_score_test = fit_evaluate(
        oracle_model,
        full_df=base_file,
        test_size=0.3,
        class_cutoff=class_cutoff,
        rating_scale=rating_scale,
    )
    print(
        f"Model selection finished! model achieved f1 score of {f1_score_test:.2f} on test_set"
    )
    print(
        f"Creating filled oracle preference matrix for sample of {num_users} users..."
    )


    candidates = base_file[USER_COL].unique().tolist()
    items = base_file[ITEM_COL].unique().tolist()

    if num_users is not None:
        idx = torch.randperm(len(candidates))[:num_users]
        users = [candidates[i] for i in idx.tolist()]

    else:
        users = candidates
    print(f"Filtering {num_users} random users and standardizing ids to idxs")
    filtered_df = base_file[base_file[USER_COL].isin(users)].copy()

    user_id_to_idx = {user: idx for idx, user in enumerate(users)}
    item_id_to_idx = {item: idx for idx, item in enumerate(items)}
    user_idx_to_id = {idx: user for user, idx in user_id_to_idx.items()}
    item_idx_to_id = {idx: item for item, idx in item_id_to_idx.items()}

    save_pickle_artifact(user_id_to_idx, get_user_id_to_idx_mapping_path(data_type, file_size, num_users))
    save_pickle_artifact(item_id_to_idx, get_item_id_to_idx_mapping_path(data_type, file_size, num_users))

    save_pickle_artifact(user_idx_to_id, get_user_idx_to_id_mapping_path(data_type, file_size, num_users))
    save_pickle_artifact(item_idx_to_id, get_item_idx_to_id_mapping_path(data_type, file_size, num_users))



    filtered_df[USER_COL] = filtered_df[USER_COL].map(user_id_to_idx)
    filtered_df[ITEM_COL] = filtered_df[ITEM_COL].map(item_id_to_idx)

    # Create oracle matrix... shape: (sampled_num_users x num_items)
    _ = get_or_create_oracle_matrix(
        oracle_model=trained_model,
        df=base_file,
        data_type=data_type,
        file_size=file_size,
        users=users,
    )
    print("Defining user timestamp behaviour from source file...")
    if data_type == "globo":
        # Handles timestamp duplicates due to k-random negative sampling
        # process: negative samples inherit the timestamp of positive samples.
        # so we remove the negative rows in order to don't affect how we calculate timestamps
        filtered_df = filtered_df[filtered_df["rating"] == 1]

    print(f"base_file - min users: {base_file[USER_COL].min()}, max users: {base_file[USER_COL].max()}, unique users: {base_file[USER_COL].nunique()}")
    print(f"filtered_df - min users: {filtered_df[USER_COL].min()}, max users: {filtered_df[USER_COL].max()}, unique users: {filtered_df[USER_COL].nunique()}")
    get_or_create_time_diff_df(filtered_df, data_type, file_size, num_users)

    print("All done!")


if __name__ == "__main__":
    main()
