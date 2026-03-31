import torch
from scipy.stats import expon

import pandas as pd

from dynamicTasteDistortion.simulationConstants import USER_COL, ITEM_COL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)


def update_genre_affinity_tensor(
    users, genre_affinity, interacted_items_tensor, item_genre_tensor
):
    # (n_interactions, n_genres)
    genre_tensor = item_genre_tensor[interacted_items_tensor].to(genre_affinity.device)

    # Scatter-add each interaction's genres into the corresponding user row
    genre_affinity.scatter_add_(
        0, users.unsqueeze(1).expand_as(genre_tensor), genre_tensor.float()
    )
    # Renormalize each user's row
    genre_affinity[users] = genre_affinity[users] / genre_affinity[users].sum(
        dim=1, keepdim=True
    ).clamp(min=1)
    # G_{u,i} = max genre affinity across item's genres: (n_users, n_items)
    G = torch.max(
        genre_affinity.unsqueeze(1) * item_genre_tensor.unsqueeze(0), dim=2
    ).values
    return genre_affinity, G


def get_item_to_genre_tensor(item_2_genre_dict):

    all_genres = sorted(
        list(set(g for genres in item_2_genre_dict.values() for g in genres))
    )
    n_genres = len(all_genres)
    n_items = len(item_2_genre_dict)

    item_genre_matrix = torch.zeros(n_items, n_genres, device=device)
    genre_name_to_id = {genre: idx for idx, genre in enumerate(all_genres)}

    for item_id, genres in item_2_genre_dict.items():
        for g in genres:
            genre_id = genre_name_to_id[g]
            item_genre_matrix[item_id, genre_id] = 1

    return item_genre_matrix


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
    """
        Looks up relevance labels for predicted items.
        Args:
            oracle_tensor: Binary relevance matrix. torch.tensor, (n_users, n_items)
            prediction_tensor: Top-k item indices per user. torch.tensor (n_users, k)
        Returns:
            Binary torch.tensor (n_users, k) — 1 if predicted item is relevant, 0 otherwise.
    """

    indices = torch.arange(
        prediction_tensor.size(0), device=prediction_tensor.device
    ).unsqueeze(1)
    return oracle_tensor[indices, prediction_tensor].int()

def update_oracle_from_hits(oracle_tensor, prediction_tensor, updated_hit_matrix):
    """
    Propagates updated hit labels back to the original oracle tensor.
    Args:
        oracle_tensor: Binary relevance matrix (n_users, n_items)
        prediction_tensor: Top-k item indices per user (n_users, k)
        updated_hit_matrix: Updated relevance labels (n_users, k)
    Returns:
        Updated oracle tensor (n_users, n_items)
    """
    oracle_updated = oracle_tensor.detach().clone()
    indices = torch.arange(prediction_tensor.size(0), device=prediction_tensor.device).unsqueeze(1)
    oracle_updated[indices, prediction_tensor] = updated_hit_matrix
    return oracle_updated

def get_user_interactions_from_predictions(predictions, hit_matrix):
    """
    Given a tensor of predictions from a Recommender and an oracle matrix that tells
    what item is relevant to each user, returns which item was actually clicked by the simulated
    user. We use a simple click model to simulate an examination, and the hit_matrix to
    simulate relevancy. If an item is relevant and examined, it was clicked.

    predictions: torch.tensor of shape (n_users, k) where each entry is a recommended
    item id to a given user.
    hit_matrix: binary torch.tensor of shape (n_users, k).
        hit_matrix[u, i] = 1 if item i is relevant to user u; 0 otherwise.



    returns
        feedback_matrix: torch.tensor of shape (n_users, k) where each entry details if the item
        was clicked by the user or not.
            feedback_matrix[u, i] > 0: item i was clicked by the user u
            feedback_matrix[u, i] < 0: item i was examined by the user u, but not clicked
            feedback_matrix[u, i] == 0: item i was not examined nor clicked.
    """

    examined_matrix = click_model(predictions)

    # click_matrix[u,i] = 1 if user examined and if recommendation was a hit
    # (is relevant); 0 otherwise.
    click_matrix = hit_matrix & examined_matrix

    # Maps a {0, 1} tensor to {1, -1} set: this is done to flag
    # un-clicked item ids as negative; clicked items have a positive id.
    should_click = 2 * click_matrix - 1

    # Interacted items are flagged as a positive item id.
    # Un-interacted items are set to a negative item id.
    interaction = should_click * predictions

    # Finally, unexaminated items are set as zero. clicked items are unchanged,
    # while examined but irrelevant items have a negative item id
    feedback_matrix = interaction * examined_matrix
    return feedback_matrix


def normalize_timestamps_per_user(recency_matrix):
    min_val = recency_matrix.min(dim=1, keepdim=True).values
    max_val = recency_matrix.max(dim=1, keepdim=True).values
    normalized = (recency_matrix - min_val) / (max_val - min_val).clamp(min=1)
    return normalized


def get_forget_probability(interaction_recency_matrix, G):
    
    # We normalize timestamps so that scale doesnt matter when dealing with forgetting.
    normalized_interaction_recency_matrix = normalize_timestamps_per_user(
        interaction_recency_matrix
    )
    # Higher G -> slower decay. So we flip G]
    exponent = normalized_interaction_recency_matrix + (1 - G)
    forget_probability = 1 - torch.exp(-exponent)
    return forget_probability


def get_users_most_recent_interaction_timestamp(interaction_recency_matrix, user):
    user_interactions = interaction_recency_matrix[user, :]
    return user_interactions.max().item()


def build_interaction_timestamp_matrix(
    n_users, n_items, oracle_tensor, initial_timestamp, dev=device
):
    """
    Build a matrix of interaction timestamps for users and items.

    Args:
        n_users (int): Number of users.
        n_items (int): Number of items.
        oracle_tensor (torch.Tensor): Tensor containing user-item interaction pairs.
        initial_timestamp (float): The timestamp value to assign to interactions.
        dev (torch.device): Device to place the tensor on (default: device).

    Returns:
        torch.Tensor: A matrix of shape (n_users, n_items) with timestamps at interaction positions.
    """
    interaction_recency_matrix = torch.zeros(
        (n_users, n_items), device=dev, dtype=torch.float32
    )
    users_with_interactions, items_interacted_with = (
        oracle_tensor[:, 0],
        oracle_tensor[:, 1],
    )

    interaction_recency_matrix[users_with_interactions, items_interacted_with] = (
        initial_timestamp
    )

    return interaction_recency_matrix


def build_interaction_matrix(predictions, hit_matrix):
    """
    Given a recommendation prediction

    predictions: Description
    hit_matrix: Description
    """
    intearction_matrix_raw = get_user_interactions_from_predictions(
        predictions, hit_matrix
    )
    interaction_matrix = torch.where(
        intearction_matrix_raw == 0,
        torch.tensor(float("nan"), device=intearction_matrix_raw.device),
        torch.where(
            intearction_matrix_raw < 0,
            torch.tensor(0, device=intearction_matrix_raw.device),
            torch.tensor(1, device=intearction_matrix_raw.device),
        ),
    )
    return interaction_matrix


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


def update_preference_matrix(
    preference_matrix,
    interaction_matrix,
    recommendation_list,
    preference_forgetting_probability,
    preference_update_rate=0.2,
):
    """
    Simulates preference acquisition and forgetting for a single timestep.
    - Acquisition: unpreferred but examined items may be liked with probability preference_update_rate.
    - Forgetting: preferred but unexamined items may be forgotten with probability preference_forgetting_probability.
    Args:
        preference_matrix: Binary preference matrix (n_users, k)
        examination_matrix: Binary matrix indicating examined items (n_users, k)
        preference_forgetting_probability: Per (user, item) forgetting probability (n_users, n_items)
        preference_update_rate: Probability of acquiring a new preference (default: 0.2)
    Returns:
        Updated binary preference matrix (n_users, k)
    """
    # Constains which items were interacted with.
    # We flip the interaction matrix, yielding which items were examined, but not clicked
    examination_matrix = 1 - interaction_matrix
    assert (
        preference_matrix.shape == examination_matrix.shape
    ), f"Shape mismatch between preference matrix {preference_matrix.shape } and examination_matrix {examination_matrix.shape}"
    preference_matrix_updated = preference_matrix.detach().clone()

    # Preferences to acquire: Relevant items that were recommended, but not interacted with.
    preferences_to_acquire = (preference_matrix_updated == 0) & (
        examination_matrix == 1
    )

    users, rec_positions = torch.where(preferences_to_acquire)

    # We update the preferences of the selected user, item pairs.
    # A taste will be acquired following a bernoulli distribution
    # with preference_update_rate probability.
    updated_preferences = torch.bernoulli(
        torch.full_like(
            preference_matrix_updated[users, rec_positions],
            preference_update_rate,
            dtype=torch.float64,
        )
    ).to(torch.int64)
    preference_matrix_updated[users, rec_positions] = updated_preferences

    # Preferences to forget: items that were not clicked by the users
    # but are relevant to them.
    preferences_to_forget = (preference_matrix_updated == 1) & (examination_matrix == 0)

    users, rec_positions = torch.where(preferences_to_forget)
    items = recommendation_list[users, rec_positions]
    # We set a preference forgetting probability per user, item pair. This is proportional
    # to the age of the last recorded interaction and the user preference for the genres
    # in item.
    preference_forgetting_rate = preference_forgetting_probability[users, items]

    # Preference forgetting rate already is a probability tensor, so we pass it to torch.bernoulli.
    updated_preferences = torch.bernoulli(preference_forgetting_rate).to(torch.int64)

    # We flip the sampled tensor because torch.bernoulli sets 1 to each entry with a probability of preference_forgetting_rate, and 0 otherwise.
    # We want to set 0 to each entry with a probability of preference_forgetting_rate, and 1 otherwise.
    # Each entry in preference_matrix_updated[users, items] == 1 by definition. So we set 0 to them
    # by flipping the updated_preferences tensor
    preference_matrix_updated[users, rec_positions] = 1 - updated_preferences

    return preference_matrix_updated


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


def random_rec(candidates, n_users, k, mask=None):
    if mask is not None:
        weights = (mask == 1).float()
        # Previously seen items are assigned a probabilty of 0 to be recommended,
        # while unseen items have a probability of 1.
        ids = torch.multinomial(weights, num_samples=k, replacement=False)
    else:
        ids = torch.randint(
            size=(n_users, k), low=0, high=len(candidates), device=device
        )

    scores = torch.rand(size=(n_users, k), device=device)
    return ids, scores
