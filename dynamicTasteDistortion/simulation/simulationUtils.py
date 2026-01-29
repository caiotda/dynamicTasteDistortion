import torch
from scipy.stats import expon

import pandas as pd

from dynamicTasteDistortion.simulationConstants import USER_COL, ITEM_COL
from dynamicTasteDistortion.simulation.tensorUtils import pandas_df_to_sparse_tensor


from calibratedRecs.calibration import Calibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)


#  TODO: refactor as a method in Calibration class
def rerank_with_calib(
    click_df, n_users, n_items, users, rec, scores, calibration_params
):

    weight = calibration_params.get("weight", "linear_time")
    distribution_mode = calibration_params.get("distribution_mode", "steck")
    _lambda = calibration_params.get("lambda", 0.99)
    device = users.device
    rec_df = pd.DataFrame(
        zip(users.tolist(), rec.tolist(), scores.tolist()),
        columns=[USER_COL, "top_k_rec_id", "top_k_rec_score"],
    ).explode(["top_k_rec_id", "top_k_rec_score"])
    history = click_df[click_df["clicked_at"] != -1.0]
    calibrator = Calibration(
        ratings_df=history,
        recommendation_df=rec_df,
        weight=weight,
        distribution_mode=distribution_mode,
        _lambda=_lambda,
        n_users=n_users,
        n_items=n_items,
    )
    calibrator.calibrate_for_users()
    reranked_df = calibrator.calibration_df
    reranked_df[ITEM_COL] = reranked_df[ITEM_COL].astype(int)
    reranked_df["rating"] = reranked_df["rating"].astype(float)
    reranked_df = reranked_df.groupby(USER_COL).agg({ITEM_COL: list, "rating": list})
    # Convert item and score numpy arrays to float type to avoid issues with torch tensor conversion
    items = reranked_df[ITEM_COL].tolist()
    scores = reranked_df["rating"].tolist()
    rec = torch.tensor(items).to(device)
    score = torch.tensor(scores).to(device)

    return rec, score


def get_user_preferences(oracle_matrix):
    user_ids = oracle_matrix[USER_COL].unique()
    item_ids = oracle_matrix[ITEM_COL].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # standardize indices
    user_indices = oracle_matrix[USER_COL].map(user_id_to_idx).values
    item_indices = oracle_matrix[ITEM_COL].map(item_id_to_idx).values
    ratings = oracle_matrix["rating"].values

    matrix = torch.zeros(
        (len(user_ids), len(item_ids)), dtype=torch.float32, device=device
    )
    # Set the relevancy of each user x item pair
    matrix[user_indices, item_indices] = (
        torch.from_numpy(ratings).to(torch.float32).to(device)
    )
    return matrix


def map_prediction_to_preferences(oracle_tensor, prediction_tensor):

    indices = torch.arange(
        prediction_tensor.size(0), device=prediction_tensor.device
    ).unsqueeze(1)
    return oracle_tensor[indices, prediction_tensor].int()


def click_model(predictions):
    """
    Simulates a click model tensor of predictions.

    Args:
        predictions (torch.tensor.int): Tensor of predictions made by the model

    Returns:
        torch.tensor.int: Returns the positions that have been examined

    Notes:
        The probability of examination is determined by a logarithmic decay function,
        where higher-ranked items have a higher chance of being examined.
    """
    M, K = predictions.shape
    # Creates a tensor of item positions in the recommendation from 0 to k,
    # for M users.
    tensor = torch.stack([torch.arange(K, device=device)] * M).to(device)
    # A random examination probability that each user has for each item position.
    examination_probability = torch.rand(M, K, device=device)
    lambda_tensor = 1 / torch.log2(tensor + 1)
    return (lambda_tensor > examination_probability).int()


def get_feedback_for_predictions(oracle_matrix, predictions):
    oracle_tensor = pandas_df_to_sparse_tensor(oracle_matrix)
    preferences_matrix = map_prediction_to_preferences(oracle_tensor, predictions)
    examined_matrix = click_model(predictions)

    should_click = 2 * (preferences_matrix & examined_matrix) - 1

    interaction = should_click * predictions
    feedback_matrix = interaction * examined_matrix

    mapped_feedback = torch.where(
        feedback_matrix == 0,
        torch.tensor(float("nan"), device=feedback_matrix.device),
        torch.where(
            feedback_matrix < 0,
            torch.tensor(0, device=feedback_matrix.device),
            torch.tensor(1, device=feedback_matrix.device),
        ),
    )

    return mapped_feedback


def get_candidate_items(D):

    user_item_pairs = D[[USER_COL, ITEM_COL]].drop_duplicates()
    user_item_matrix = (
        user_item_pairs.assign(interaction=-1)
        .pivot(index=USER_COL, columns=ITEM_COL, values="interaction")
        .fillna(1)
        .astype(int)
    )

    mask_from_df = torch.tensor(
        user_item_matrix.values, dtype=torch.int8, device=device
    )
    return mask_from_df


def random_rec(candidates, n_users, k):
    ids = torch.randint(size=(n_users, k), low=0, high=len(candidates), device=device)
    scores = torch.rand(size=(n_users, k), device=device)
    return ids, scores
