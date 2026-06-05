input_size_to_file_name = {
    "s": "1m",
    "m": "10m",
    "l": "20m",
}


GLOBO_PATH = "dynamicTasteDistortion/data/globo"
MOVIELENS_PATH = "dynamicTasteDistortion/data/movielens"
YELP_PATH = "dynamicTasteDistortion/data/yelp"
FOOD_PATH = "dynamicTasteDistortion/data/food"

ARTIFACTS_PATH = "dynamicTasteDistortion/artifacts"
RESULTS_PATH = f"{ARTIFACTS_PATH}/results"
MODEL_ARTIFACTS_PATH = f"{ARTIFACTS_PATH}/model"

SIMULATION_PATH = "dynamicTasteDistortion/data/simulation"

USER_COL = "user"
ITEM_COL = "item"
GENRES_COL = "genres"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"

REVIEWS_PER_USER_THRESHOLD = 30

# Seeds generated via random.sample(range(1000), 20)
# Except for the first one, which we set to 42
# to keep results comparable to a previous version of the code
# that didn't repeat experiments several times.
SEEDS = [
    42,
    86,
    840,
    394,
    806,
    558,
    426,
    768,
    208,
    436,
    590,
    98,
    41,
    833,
    62,
    472,
    645,
    905,
    991,
    698,
]
