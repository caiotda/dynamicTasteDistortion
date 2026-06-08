import numpy as np
import pandas as pd
from surprise import Reader, Dataset as SurpriseDataset
from tqdm import tqdm


from dynamicTasteDistortion.simulationConstants import ITEM_COL, USER_COL

import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

def fill_out_matrix(base_df, model, user_sample, item_id_to_idx_map, user_id_to_idx_map, rating_cutoff=4.0, batch_size=50_000, rating_scale=(1,5)):
    # ACM Disclosure: This function was AI generated. The previous version, which can be found
    # at this hash command (link) worked, but was prone to OOM errors. This new version uses 
    # pyarrow to save partial results in disk and uses parquet files instead of csv, avoiding 
    # RAM overhead.

    reader = Reader(rating_scale=rating_scale)
    trainset = SurpriseDataset.load_from_df(
        base_df[["user", "item", "rating"]], reader
    ).build_full_trainset()
    # Trains model on entire dataset, no filtering: avoids missing important neighborhood information
    model.fit(trainset)
    # We are only interested in generating predictions for a subset of users.
    # But we want to predict for every item in the original dataset.
    user_sample = list(user_sample)

    all_items = trainset.all_items()
    # user_ids = [trainset.to_raw_uid(u) for u in all_users]
    item_ids = [trainset.to_raw_iid(i) for i in all_items]

    genres_map = base_df[["item", "genres"]].drop_duplicates("item").set_index("item")["genres"]

    schema = pa.schema([
        pa.field("user", pa.string()),
        pa.field("item", pa.string()),
        pa.field("rating", pa.int8()),
        pa.field("genres", pa.list_(pa.string())),
    ])

    tmp_dir = tempfile.mkdtemp()
    writer = pq.ParquetWriter(os.path.join(tmp_dir, "predictions.parquet"), schema)

    buffer = []
    def flush():
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            writer.write_table(table)
            buffer.clear()

    for user_id in tqdm(user_sample, desc="Predicting ratings for sampled users..."):
        inner_uid = trainset.to_inner_uid(user_id)
        rated_items = {trainset.to_raw_iid(i) for i, _ in trainset.ur[inner_uid]}

        for item_id in item_ids:
            if item_id not in rated_items:
                # We need to predict with the original user id, hence 
                # why we map it back. item_id is already the original
                # id.
                est = model.predict(user_id, item_id).est
                # We store the idx version of each user and item idx.
                # The result is a standardized dataset.
                buffer.append({
                    "user": str(user_id_to_idx_map[user_id]),
                    "item": str(item_id_to_idx_map[item_id]),
                    "rating": int(est >= rating_cutoff),
                    "genres": genres_map.get(item_id, ""),
                })
                if len(buffer) >= batch_size:
                    flush()
    flush() 

    writer.close()

    
    # Merge the filtered dataset with the filled out version.
    filtered_original_df = base_df[base_df[USER_COL].isin(user_sample)].copy()
    filtered_original_df[USER_COL] = filtered_original_df[USER_COL].map(user_id_to_idx_map)
    filtered_original_df[ITEM_COL] = filtered_original_df[ITEM_COL].map(item_id_to_idx_map)
    base_out = filtered_original_df[["user", "item", "genres", "rating"]].copy()
    base_out["user"] = base_out["user"].astype(str)
    base_out["item"] = base_out["item"].astype(str)
    base_out["rating"] = (base_out["rating"] >= rating_cutoff).astype("int8")

    base_table = pa.Table.from_pandas(base_out, schema=schema, preserve_index=False)
    base_writer = pq.ParquetWriter(os.path.join(tmp_dir, "base.parquet"), schema)
    base_writer.write_table(base_table)
    base_writer.close()

    df_filled = pd.read_parquet(tmp_dir)
    df_filled["user"] = df_filled["user"].astype(int)
    df_filled["item"] = df_filled["item"].astype(int)
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
