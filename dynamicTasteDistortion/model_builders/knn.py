from recmodels.knn import Uknn, Iknn


def build_iknn(config, hyperparameter_tuning_df):
    print(config)
    k_neighbors = config["model_params"]["k_neighbors"]
    return Iknn(df=hyperparameter_tuning_df, k_neighbors=k_neighbors)


def build_uknn(config, hyperparameter_tuning_df):
    k_neighbors = config["model_params"]["k_neighbors"]
    return Uknn(df=hyperparameter_tuning_df, k_neighbors=k_neighbors)
