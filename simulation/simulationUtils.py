import math
import random
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np

from tasteDistortionOnDynamicRecs.simulationConstants import USER_COL, ITEM_COL
from tasteDistortionOnDynamicRecs.simulation.tensorUtils import get_matrix_coordinates

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=42
torch.manual_seed(seed)


def get_user_preferences(oracle_matrix):
    user_ids = oracle_matrix[USER_COL].unique()
    item_ids = oracle_matrix[ITEM_COL].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # standardize indices
    user_indices = oracle_matrix[USER_COL].map(user_id_to_idx).values
    item_indices = oracle_matrix[ITEM_COL].map(item_id_to_idx).values
    ratings = oracle_matrix["rating"].values

    matrix = torch.zeros((len(user_ids), len(item_ids)), dtype=torch.float32, device=device)
    # Set the relevancy os each user x item pair
    matrix[user_indices, item_indices] = torch.from_numpy(ratings).to(torch.float32).to(device)
    return matrix

def map_prediction_to_preferences(oracle_tensor, prediction):
    item_idx_tensor = prediction

    U = oracle_tensor.shape[0]
    indices = torch.arange(U).unsqueeze(1).to(device)


    return oracle_tensor[indices, item_idx_tensor].int()

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
    lambda_tensor = 1/torch.log2(tensor+1)
    return (lambda_tensor > examination_probability).int()


def get_feedback_for_predictions(oracle_matrix, predictions):
    oracle_tensor = get_user_preferences(oracle_matrix)
    preferences_matrix = map_prediction_to_preferences(oracle_tensor, predictions)
    examined_matrix = click_model(predictions)

    should_click = 2*(preferences_matrix & examined_matrix) - 1

    interaction = should_click * predictions
    feedback_matrix = interaction * examined_matrix

    mapped_feedback = torch.where(
        feedback_matrix == 0,
        torch.tensor(float('nan'), device=feedback_matrix.device),
        torch.where(
            feedback_matrix < 0,
            torch.tensor(0, device=feedback_matrix.device),
            torch.tensor(1, device=feedback_matrix.device)
        )
    )


    return mapped_feedback
    

def get_candidate_items(D):

    user_item_pairs = D[[USER_COL, ITEM_COL]].drop_duplicates()
    user_item_matrix = user_item_pairs.assign(interaction=-1).pivot(index=USER_COL, columns=ITEM_COL, values='interaction').fillna(1).astype(int)

    mask_from_df = torch.tensor(user_item_matrix.values, dtype=torch.int8, device=device)
    return mask_from_df


def random_rec(candidates, n_users, k):
    return torch.randint(
        size=(n_users, k),
        low=0,
        high=len(candidates),
        device=device
    )

def simulate_user_feedback(users, candidate_items, mask, oracle_matrix, k, rating_delta_distribution, model,user_idx_to_id_map, initial_time=0.0, feedback_from_bootstrap=False):
    """
        Simulates user feedback for a batch of users by recommending k items and mapping the recommendations to feedback.

        Args:
            users (torch.Tensor): Tensor of user indices.
            candidate_items (torch.Tensor): Tensor of candidate items for recommendation.
            mask (torch.Tensor): 2D tensor indicating items to ignore during recommendation (i.e: previously interacted items).
            oracle_matrix (pd.DataFrame): Oracle preference matrix dataframe.
            k (int): Number of items to recommend per user.
            rating_delta_distribution (dict): Dictionary mapping user indices to their corresponding
                                              exponential distribution for time between interactions.
            model (object): Recommendation model with a `predict` method that takes users, k, candidates, and mask.
            initial_time (float, optional): Initial timestamp for the simulation. Defaults to 0.0.
            feedback_from_bootstrap (bool, optional): If True, generates random recommendations instead of using the model. Defaults to False.

        Returns:
            pd.DataFrame: A dataframe containing the simulated interactions with the following schema:
                - users: User indices.
                - items: Recommended item indices.
                - feedback: Feedback values (1 for positive, 0 for negative, NaN for no interaction).
                - clicked_at: Click positions in the recommendation list (NaN if no click occurred).
                - timestamp: Interaction timestamps (NaN if no interaction occurred).
    """
    if (feedback_from_bootstrap):
        n_users = len(users)
        rec = random_rec(candidate_items, n_users, k)
    else:   
        rec = model.predict(user=users, k=k, candidates=candidate_items, mask=mask)[0]

    feedback_matrix = get_feedback_for_predictions(oracle_matrix, rec)


    indices =  get_matrix_coordinates(feedback_matrix)

    users_indices, click_positions = indices[:, 0].tolist(), indices[:, 1].tolist()
    user_ids = [user_idx_to_id_map[idx] for idx in users_indices]

    feedbacks = feedback_matrix.flatten().tolist()
    items = rec.flatten().tolist()

    timestamps = [(rating_delta_distribution[user].rvs(1)[0] / 60) + initial_time for user in user_ids]
    entries = list(zip(users_indices, items, feedbacks, click_positions, timestamps))
    interaction_df = pd.DataFrame(entries, columns=["user", "item", "feedback", "clicked_at", "timestamp"])


    interaction_df.loc[interaction_df["feedback"] != 1.0, "clicked_at"] = np.nan
    interaction_df.loc[interaction_df["feedback"] != 1.0, "timestamp"] = np.nan

    return interaction_df