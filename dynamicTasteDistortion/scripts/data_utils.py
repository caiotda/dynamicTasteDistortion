import unicodedata
import re

from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
    REVIEWS_PER_USER_THRESHOLD,
)


synonyms = {
    "children": "child",
    "childs": "child",
    "childrens": "child",
    "thrill": "thriller",
}


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
