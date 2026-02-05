import pandas as pd

from dynamicTasteDistortion.scripts.data_utils import (
    filter_inactive_users,
    preprocess_genres,
)

pd.options.mode.chained_assignment = None

import unicodedata
import wget
import zipfile
import os
import re
import argparse
import json

import math

from dynamicTasteDistortion.simulationConstants import (
    MOVIELENS_PATH,
    REVIEWS_PER_USER_THRESHOLD,
    STEAM_PATH,
    YELP_PATH,
    USER_COL,
    ITEM_COL,
    GENRES_COL,
    RATING_COL,
    input_size_to_file_name,
)

input_size_to_sample_size = {
    "s": 1_000_000,
    "m": 10_000_000,
    "l": 20_000_000,
}


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


def read_steam_sub_file(file_name):
    return pd.read_csv(f"{STEAM_PATH}/raw/{file_name}.csv")


def read_steam_raw(size):
    steam_url = "https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam"
    destination_dir = f"{STEAM_PATH}/raw/steam_{size}/"
    _ = download(steam_url, destination_dir)
    print("Reading steam sub files...")
    reviews = read_steam_sub_file("recommendations")[
        ["user_id", "app_id", "is_recommended"]
    ]
    users = read_steam_sub_file("users")
    active_users = users[users["reviews"] > REVIEWS_PER_USER_THRESHOLD][
        "user_id"
    ].tolist()
    filtered_reviews = reviews[reviews["user_id"].isin(active_users)]

    games = read_json_file(destination_dir, "games_metadata.json")
    games["num_tags"] = games["tags"].apply(len)
    filtered_games = games[games["num_tags"] > 0]
    filtered_games["genres"] = filtered_games["tags"].apply(lambda l: ",".join(l))
    unsampled_steam_df = filtered_reviews.merge(filtered_games, on="app_id")[
        ["user_id", "app_id", "genres", "is_recommended"]
    ]

    original_size = unsampled_steam_df.shape[0]
    if size > original_size:
        sample_size = original_size
        print(f"Requested sample size larger than dataset! Defaulting to full dataset.")
    else:
        sample_size = size

    steam_df = unsampled_steam_df.sample(n=sample_size)
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
    yelp_url = (
        "https://www.kaggle.com/api/v1/datasets/download/yelp-dataset/yelp-dataset"
    )
    destination_dir = f"{YELP_PATH}/raw/yelp_{size}/"
    _ = download(yelp_url, destination_dir)
    review_file = "yelp_academic_dataset_review.json"
    user_file = "yelp_academic_dataset_user.json"
    business_file = "yelp_academic_dataset_business.json"

    users_df = read_json_file(destination_dir, user_file, limit=size)[
        ["user_id", "review_count"]
    ]
    filtered_users_df = users_df[users_df["review_count"] >= REVIEWS_PER_USER_THRESHOLD]
    users_to_keep = list(filtered_users_df["user_id"].unique())

    reviews_df = read_json_file(destination_dir, review_file, limit=size)[
        ["user_id", "business_id", "stars", "date"]
    ]
    filtered_reviews = reviews_df[reviews_df["user_id"].isin(users_to_keep)]

    business_df = read_json_file(destination_dir, business_file, limit=size)[
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
    destination_dir = f"{MOVIELENS_PATH}/raw/ml_{size}/"

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


def process_ml_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")
    processed_df = df.copy()

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
    processed_df = df.copy()

    processed_df[GENRES_COL] = preprocess_genres(processed_df, GENRES_COL, SEP=",")
    processed_df["binarized_rating"] = processed_df[RATING_COL].apply(
        lambda boolean_rating: int(boolean_rating)
    )
    return processed_df


def process_yelp_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")
    processed_df = df.copy()

    # Padronizar a coluna de generos
    processed_df = processed_df[~processed_df[GENRES_COL].isna()]
    processed_df[GENRES_COL] = preprocess_genres(processed_df, GENRES_COL, SEP=",")
    # ratings >= 4 -> 1 (binarized)
    processed_df["binarized_rating"] = processed_df[RATING_COL].apply(
        lambda rating: int(rating >= 4)
    )
    processed_df["timestamp"] = (
        pd.to_datetime(processed_df["date"]).astype("int64") // 10**9
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
        choices=["s", "m", "l"],
        required=True,
        help="Dataset size: s (1m), m (10m), l (20m). If Sample size is larger than dataset, then everything is returned",
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
        if size > df.shape[0]:
            output_file_size = "full"
        output_file = f"{STEAM_PATH}/steam_{output_file_size}"

    df.to_csv(f"{output_file}.csv", index=False)
    df.to_pickle(f"{output_file}.pkl")
    print(f"Processed dataset saved to path {output_file}")


if __name__ == "__main__":
    main()
