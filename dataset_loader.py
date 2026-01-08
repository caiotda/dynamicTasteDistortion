import pandas as pd

pd.options.mode.chained_assignment = None

import unicodedata
import wget
import zipfile
import os
import re
import argparse
import json

import math

from simulationConstants import (
    MOVIELENS_PATH,
    STEAM_PATH,
    YELP_PATH,
    USER_COL,
    ITEM_COL,
    GENRES_COL,
    RATING_COL,
)

input_size_to_file_name = {
    "xs": "100k",
    "s": "1m",
    "m": "10m",
    "l": "20m",
    "xl": "full",
}
input_size_to_sample_size = {
    "xs": 100_00,
    "s": 1_000_000,
    "m": 10_000_000,
    "l": 20_000_000,
    "xl": math.inf,
}


synonyms = {
    "children": "child",
    "childs": "child",
    "childrens": "child",
    "thrill": "thriller",
}


REVIEWS_PER_USER_THRESHOLD = 30


def get_ml_url(size):
    file_size = input_size_to_file_name[size]
    return f"https://files.grouplens.org/datasets/movielens/ml-{file_size}.zip"


def download(dataset_url, destination_dir):
    if os.path.exists(destination_dir) and os.listdir(destination_dir):
        print(f"Skipping: '{destination_dir}' already exists and contains files.")
        return destination_dir
    print("Downloading file...")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    file_name = os.path.basename(dataset_url)
    if os.path.exists(file_name):
        os.remove(file_name)
        # Remove all .tmp files in the current folder
        for f in os.listdir("."):
            if f.endswith(".tmp"):
                os.remove(f)
    file_name = wget.download(dataset_url, f"{file_name}.zip")
    print("Unzipping...")
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    # Delete the .zip file after extraction
    if os.path.exists(file_name):
        os.remove(file_name)
    return file_name


def download_yelp_files():
    yelp_url = (
        "https://www.kaggle.com/api/v1/datasets/download/yelp-dataset/yelp-dataset"
    )
    destination_dir = f"{YELP_PATH}/raw/"
    _ = download(yelp_url, destination_dir)
    return destination_dir


def download_steam_file():
    steam_url = "https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam"
    destination_dir = f"{STEAM_PATH}/raw/"
    _ = download(steam_url, destination_dir)
    return destination_dir


def read_steam_sub_file(file_name):
    return pd.read_csv(f"{STEAM_PATH}/raw/{file_name}.csv")


def read_steam_raw(size):
    steam_path = download_steam_file()
    print("Reading steam sub files...")
    reviews = read_steam_sub_file("recommendations")[
        ["user_id", "app_id", "is_recommended"]
    ]
    users = read_steam_sub_file("users")
    active_users = users[users["reviews"] > REVIEWS_PER_USER_THRESHOLD][
        "user_id"
    ].tolist()
    filtered_reviews = reviews[reviews["user_id"].isin(active_users)]

    games = read_json_file(steam_path, "games_metadata.json")
    games["num_tags"] = games["tags"].apply(len)
    filtered_games = games[games["num_tags"] > 0]
    filtered_games["genres"] = filtered_games["tags"].apply(lambda l: ",".join(l))
    steam_df = filtered_reviews.merge(filtered_games, on="app_id")[
        ["user_id", "app_id", "genres", "is_recommended"]
    ].sample(n=size)
    print("Done!")
    return steam_df.rename(
        columns={
            "user_id": USER_COL,
            "app_id": ITEM_COL,
            "genres": GENRES_COL,
            "is_recommended": RATING_COL,
        }
    )


def read_json_file(base_path, file, limit=math.inf):
    if base_path.endswith("/") is False:
        base_path += "/"
    data_file = open(f"{base_path}{file}")
    data = []
    for line_number, line in enumerate(data_file, 1):
        if line_number > limit:
            break
        data.append(json.loads(line))
    data_file.close()
    return pd.DataFrame(data)


def read_yelp_raw(size):
    yelp_path = download_yelp_files()
    review_file = "yelp_academic_dataset_review.json"
    user_file = "yelp_academic_dataset_user.json"
    business_file = "yelp_academic_dataset_business.json"

    users_df = read_json_file(yelp_path, user_file, limit=size)[
        ["user_id", "review_count"]
    ]
    filtered_users_df = users_df[users_df["review_count"] >= REVIEWS_PER_USER_THRESHOLD]
    users_to_keep = list(filtered_users_df["user_id"].unique())

    reviews_df = read_json_file(yelp_path, review_file, limit=size)[
        ["user_id", "business_id", "stars", "date"]
    ]
    filtered_reviews = reviews_df[reviews_df["user_id"].isin(users_to_keep)]

    business_df = read_json_file(yelp_path, business_file, limit=size)[
        ["business_id", "categories"]
    ].drop_duplicates()
    yelp_df = filtered_reviews.merge(business_df, on="business_id")

    return yelp_df.rename(
        columns={
            "user_id": USER_COL,
            "business_id": ITEM_COL,
            "categories": GENRES_COL,
            "stars": RATING_COL,
        }
    )


def read_ml_raw(size):
    dataset_url = get_ml_url(size)
    destination_dir = f"{MOVIELENS_PATH}/raw/"

    _ = download(dataset_url, destination_dir)
    file_name_cleaned = (
        "ml-10M100K" if size == "m" else f"ml-{input_size_to_file_name[size]}"
    )
    files_dir = destination_dir + file_name_cleaned
    file_format = "csv" if size == "l" else "dat"
    sep = "," if size == "l" else "::"
    header = 0 if size == "l" else None
    movies = pd.read_csv(
        f"{files_dir}/movies.{file_format}",
        sep=sep,
        encoding="ISO-8859-1",
        header=header,
        engine="python",
        names=["item_id", "item_name", "genres"],
    )
    ratings = pd.read_csv(
        f"{files_dir}/ratings.{file_format}",
        sep=sep,
        encoding="ISO-8859-1",
        header=header,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    print("Joining movies data with ratings data...")

    base_df = movies.merge(ratings, on="item_id").drop(columns=["item_name"])
    base_df = base_df.rename(
        columns={"user_id": USER_COL, "item_id": ITEM_COL, "genres": GENRES_COL}
    )
    return base_df


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

    return processed_df


def process_ml_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")
    processed_df = standardize_ids(df)

    # Padronizar a coluna de generos

    processed_df[GENRES_COL] = preprocess_genres(df, GENRES_COL)
    # Filtrar usuarios inativos?
    filtered_df = filter_inactive_users(processed_df)
    # ratings >= 4 -> 1 (binarized)
    filtered_df["binarized_rating"] = filtered_df[RATING_COL].apply(
        lambda rating: int(rating >= 4)
    )
    return filtered_df


def process_steam_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")
    processed_df = standardize_ids(df)

    processed_df[GENRES_COL] = preprocess_genres(processed_df, GENRES_COL, SEP=",")
    processed_df["binarized_rating"] = processed_df[RATING_COL].apply(
        lambda boolean_rating: int(boolean_rating)
    )
    return processed_df


def process_yelp_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")
    processed_df = standardize_ids(df)

    # Padronizar a coluna de generos
    processed_df = processed_df[~processed_df[GENRES_COL].isna()]
    processed_df[GENRES_COL] = preprocess_genres(processed_df, GENRES_COL, SEP=",")
    # ratings >= 4 -> 1 (binarized)
    processed_df["binarized_rating"] = processed_df[RATING_COL].apply(
        lambda rating: int(rating >= 4)
    )
    return processed_df


def get_ml_df(size):
    raw_df = read_ml_raw(size)
    return process_ml_df(raw_df)


def get_yelp_df(size):
    raw_df = read_yelp_raw(size)
    return process_yelp_df(raw_df)


def get_steam_df(size):
    raw_df = read_steam_raw(size)
    return process_steam_df(raw_df)


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets.")
    parser.add_argument(
        "--size",
        choices=["xs", "s", "m", "l", "xl"],
        required=True,
        help="Dataset size: xs (100k), s (1m), m (10m), l (20m), xl (As large as possible)",
    )
    parser.add_argument(
        "--data",
        choices=["ml", "yelp", "steam"],
        required=True,
        help="Dataset type: ml (MovieLens); yelp; steam",
    )

    args = parser.parse_args()
    size = input_size_to_sample_size[args.size]
    output_file_size = input_size_to_file_name[args.size]

    if args.data == "ml":
        size = input_size_to_file_name[args.size]
        output_file = f"{MOVIELENS_PATH}/ml_{size}"
        df = get_ml_df(args.size)

    if args.data == "yelp":
        df = get_yelp_df(size)
        output_file = f"{YELP_PATH}/yelp_{output_file_size}"

    if args.data == "steam":
        df = get_steam_df(size)
        output_file = f"{STEAM_PATH}/steam_{output_file_size}"

    df.to_csv(f"{output_file}.csv", index=False)
    df.to_pickle(f"{output_file}.pkl")
    print(f"Processed dataset saved to path {output_file}")


if __name__ == "__main__":
    main()
