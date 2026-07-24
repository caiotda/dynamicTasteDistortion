import unicodedata
import re

import numpy as np
import pandas as pd

from tqdm import tqdm
from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
    REVIEWS_PER_USER_THRESHOLD,
)

import gc
import torch

synonyms = {
    "children": "child",
    "childs": "child",
    "childrens": "child",
    "thrill": "thriller",
}


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def concat_dfs(df1, df2):
    # Wrapper to handle with annoying FutureWarning when dealing
    # with a empty dataframe
    dfs = [df1, df2]
    to_concat = [df for df in dfs if not df.empty]
    final_df = pd.concat(to_concat, ignore_index=True)
    return final_df


def sample_negatives(df, k, candidates):
    candidates_arr = np.array(list(candidates))
    observed = set(zip(df["user"], df["item"]))

    users = df["user"].values
    n = len(users)

    sampled = np.random.choice(candidates_arr, size=(n, k), replace=True)
    timestamps = df["timestamp"].values

    records = []
    for i, user in tqdm(enumerate(users), total=n, desc="Processing users..."):
        for candidate in sampled[i]:
            if (user, candidate) not in observed:
                records.append((user, candidate, 0, timestamps[i]))

    positives = df[["user", "item", "timestamp"]].copy()
    positives["rating"] = 1
    negatives = pd.DataFrame(records, columns=["user", "item", "rating", "timestamp"])

    result = pd.concat([positives, negatives]).reset_index(drop=True)
    item_genre_map = (
        df[["item", "genres"]].drop_duplicates().set_index("item")["genres"]
    )
    result["genres"] = result["item"].map(item_genre_map)
    result["genres"] = result["genres"].apply(lambda x: [str(x)])

    return result


def preprocess_genres(df, genre_col="genres", SEP="|"):
    return df[genre_col].apply(
        lambda text: [text_preprocess(token) for token in text.split(SEP)]
    )


def normalize_word(word):
    return synonyms.get(word, word)


def text_preprocess(text):
    text = normalize_word(text)
    text = text.lower()
    text = text.strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", "-", text)
    return text


def filter_inactive_users(df, threshold=REVIEWS_PER_USER_THRESHOLD):
    users_to_keep = (
        df.groupby(USER_COL)
        .agg({ITEM_COL: "count"})
        .reset_index()
        .rename(columns={ITEM_COL: "n_reviews"})
        .query(f"n_reviews >= {threshold}")[USER_COL]
    )
    users_to_keep = list(set(users_to_keep))
    filtered_df = df[df[USER_COL].isin(users_to_keep)]

    return filtered_df


def standardize_ids(df):

    processed_df = df.copy()

    unique_user_ids = df[USER_COL].unique()
    unique_item_ids = df[ITEM_COL].unique()
    user_id_map = {
        old_id: new_id for new_id, old_id in enumerate(sorted(unique_user_ids))
    }
    item_id_map = {
        old_id: new_id for new_id, old_id in enumerate(sorted(unique_item_ids))
    }
    processed_df[USER_COL] = processed_df[USER_COL].map(user_id_map)
    processed_df[ITEM_COL] = processed_df[ITEM_COL].map(item_id_map)

    return processed_df, user_id_map, item_id_map
