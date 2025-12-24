import pandas as pd

pd.options.mode.chained_assignment = None

import unicodedata
import wget
import zipfile
import os
import re
import argparse


from simulationConstants import MOVIELENS_PATH, YELP_PATH, USER_COL, ITEM_COL, GENRES_COL

size_to_file_name = {"s": "1m", "m": "10m", "l": "20m"}


synonyms = {
    "children": "child",
    "childs": "child",
    "childrens": "child",
    "thrill": "thriller",
}


REVIEWS_PER_USER_THRESHOLD = 30


def get_ml_url(size):
    file_size = size_to_file_name[size]
    return f"https://files.grouplens.org/datasets/movielens/ml-{file_size}.zip"


def download(dataset_url, destination_dir):
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


def read_yelp_ml_raw():
    yelp_url = (
        "https://www.kaggle.com/api/v1/datasets/download/yelp-dataset/yelp-dataset"
    )
    destination_dir = f"{YELP_PATH}/raw/"
    file_name = download(yelp_url, destination_dir)
    return file_name


def read_ml_raw(size):
    dataset_url = get_ml_url(size)
    destination_dir = f"{MOVIELENS_PATH}/raw/"

    file_name = download(dataset_url, destination_dir)
    file_name_cleaned = "ml-10M100K" if size == "m" else f"ml-{size_to_file_name[size]}"
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
        lambda text: [
            normalize_word(re.sub(r"[^a-zA-Z0-9\s]", "", token))
            for token in text.split(SEP)
        ]
    )


def normalize_word(word):
    return synonyms.get(word, word)


def text_preprocess(text):
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


def process_ml_df(df):
    # Padronizar user id e item id
    print("Preprocessing dataset...")

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

    # Padronizar a coluna de generos

    processed_df[GENRES_COL] = preprocess_genres(df, GENRES_COL)
    # Filtrar usuarios inativos?
    filtered_df = filter_inactive_users(processed_df)
    # ratings >= 4 -> 1 (binarized)
    filtered_df["binarized_rating"] = filtered_df["rating"].apply(
        lambda rating: int(rating >= 4)
    )
    return filtered_df


def get_ml_df(size):
    raw_df = read_ml_raw(size)
    return process_ml_df(raw_df)


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets.")
    parser.add_argument(
        "--size",
        choices=["s", "m", "l"],
        required=True,
        help="Dataset size: s (1m), m (10m), l (20m)",
    )
    parser.add_argument(
        "--data", choices=["ml", "yelp"], required=True, help="Dataset type: ml (MovieLens)"
    )

    args = parser.parse_args()
    size = size_to_file_name[args.size]
    if args.data == "ml":
        df = get_ml_df(args.size)
        output_file = f"{MOVIELENS_PATH}/ml_{size}.csv"
        df.to_csv(output_file, index=False)
        print(f"Processed dataset saved to {output_file}")
    if args.data == "yelp":
        destination = read_yelp_ml_raw()
        print(f"Yelp dataset saved to {destination}")


if __name__ == "__main__":
    main()
