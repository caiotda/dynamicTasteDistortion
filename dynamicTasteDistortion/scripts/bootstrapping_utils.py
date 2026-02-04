import numpy as np
import pandas as pd
from surprise import Reader, Dataset as SurpriseDataset
from tqdm import tqdm


from dynamicTasteDistortion.simulationConstants import USER_COL


def fill_out_matrix(base_df, model, user_sample):
    reader = Reader(rating_scale=(1, 5))
    df = base_df[base_df[USER_COL].isin(user_sample)]
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
                pred = model.predict(user_id, item_id)
                predictions.append([user_id, item_id, pred])

    # binarize predictions
    processed_predictions = [
        [pred[0], pred[1], int(pred[2].est >= 4)] for pred in predictions
    ]

    predictions_df = pd.DataFrame(
        processed_predictions, columns=["user", "item", "rating"]
    )

    genres_df = df[["item", "genres"]].drop_duplicates(subset="item")
    predictions_df = predictions_df.merge(genres_df, on="item")
    base_df = df[["user", "item", "genres", "rating"]]
    base_df.loc[:, "rating"] = base_df["rating"].apply(lambda rating: int(rating >= 4))

    # Combine missing entries with previously filled
    df_filled = pd.concat([base_df, predictions_df], ignore_index=True)

    return df_filled


def get_timestamp_behavior(base_df, sample):
    df = base_df[base_df[USER_COL].isin(sample)]
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

    global_std_timestamp_diff = np.std(
        np.diff(
            df[df[USER_COL].isin(positive_timestamp_diff)].sort_values(
                [USER_COL, "timestamp"]
            )["timestamp"]
        )
    )

    avg_std_time_diff_per_user["median_timestamp_diff"] = avg_std_time_diff_per_user[
        "median_timestamp_diff"
    ].replace([0, np.nan], global_median_timestamp_diff)

    avg_std_time_diff_per_user["std_timestamp_diff"] = avg_std_time_diff_per_user[
        "std_timestamp_diff"
    ].replace([0, np.nan], global_std_timestamp_diff)

    return avg_std_time_diff_per_user
