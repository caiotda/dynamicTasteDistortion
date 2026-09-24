from dynamicTasteDistortion.dataset_loader import input_size_to_sample_size, load_df

from dynamicTasteDistortion.scripts.data_utils import standardize_ids
from dynamicTasteDistortion.scripts.model_utils import MostPopularRecommender


def build_most_popular(config, hyperparameter_tuning_df):
    if config["overwrite_model_selection"]:
        raise ValueError("overwrite_model_selection is not supported for most_popular.")

    data_type, size = config["data_type"], config["size"]
    file_size = input_size_to_sample_size[size]
    print(f"Loading {data_type}_{file_size} dataset to fit Most Popular model...")
    sample_size = size if data_type == "ml" else file_size
    df = load_df(data_type, size=sample_size)
    processed_df, _, _ = standardize_ids(df)
    return MostPopularRecommender(processed_df)
