input_size_to_file_name = {
    "s": "1m",
    "m": "10m",
    "l": "20m",
}


MOVIELENS_PATH = "data/movielens"


ML_DATA_PATH = f"{MOVIELENS_PATH}/movielens_1m.csv"
ML_RATINGS_DATA_PATH = f"{MOVIELENS_PATH}/movielens_1m_ratings.csv"
ML_BIN_RATINGS_DATA_PATH = f"{MOVIELENS_PATH}/movielens_1m_bin_ratings.csv"
STEAM_PATH = "data/steam"
YELP_PATH = "data/yelp"

ARTIFACTS_PATH = "artifacts"
RESULTS_PATH = f"{ARTIFACTS_PATH}/results"
MODEL_ARTIFACTS_PATH = f"{ARTIFACTS_PATH}/model"

SIMULATION_PATH = "data/simulation"
ML_1M_ORACLE_PATH = f"{SIMULATION_PATH}/movielens_1m_oracle.csv"
ML_1M_FILLED_PATH = f"{SIMULATION_PATH}/movielens_1m_sinthetically_filled.csv"
ML_1M_FILLED_PATH_PKL = f"{SIMULATION_PATH}/movielens_1m_sinthetically_filled.pkl"
ML_1M_1K_SAMPLE_FILLED_PATH = (
    f"{SIMULATION_PATH}/movielens_1m_1k_sample_sinthetically_filled.csv"
)
ML_1M_1K_SAMPLE_FILLED_PATH_PKL = (
    f"{SIMULATION_PATH}/movielens_1m_1k_sample_sinthetically_filled.pkl"
)

USER_COL = "user"
ITEM_COL = "item"
GENRES_COL = "genres"
RATING_COL = "rating"
