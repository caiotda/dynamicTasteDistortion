import torch

from typing import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def apply_rolling_avg(metric, window_size=10):
    return pd.Series(metric).rolling(window=window_size).mean()


def remove_outliers(metric):

    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(metric, 25)
    Q3 = np.percentile(metric, 75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    metric_cleaned = [x for x in metric if lower_bound <= x <= upper_bound]

    return metric_cleaned


def precompute_jaccard(genre_lookup):
    """
    Args:
        genre_lookup: Binary tensor of shape (n_items, n_genres)
    Returns:
        jaccard_matrix: Float tensor of shape (n_items, n_items)
    """
    g = genre_lookup.float()
    intersection = g @ g.T  # (n_items, n_items)
    genre_counts = g.sum(dim=1)  # (n_items,)
    union = (
        genre_counts.unsqueeze(1) + genre_counts.unsqueeze(0) - intersection
    )  # (n_items, n_items)
    return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))


def diversity(recs, jaccard_lookup):
    return 1 - intra_list_similarity(recs, jaccard_lookup)


def intra_list_similarity(
    recommendations: torch.Tensor,
    jaccard_matrix: torch.Tensor,
) -> float:
    n_users, k = recommendations.shape
    if k < 2:
        raise ValueError(
            "Need at least 2 items per user to compute pairwise similarity."
        )

    rows = recommendations.unsqueeze(2).expand(n_users, k, k).contiguous()
    cols = recommendations.unsqueeze(1).expand(n_users, k, k).contiguous()
    user_matrices = jaccard_matrix[rows, cols]

    mask = torch.ones(k, k, dtype=torch.bool).triu(diagonal=1)  # (k, k)
    pair_sims = user_matrices[:, mask]  # (n_users, n_pairs)

    return pair_sims.mean().item()


def calculate_gini_index(recommendations, catalog):
    """
    Calculate the Gini Index of a recommendation list to measure popularity bias.

    Args:
        recommendations: Integer tensor of shape (n_users, k) where each row contains
                         the k recommended item IDs for a user.
        catalog: List of all possible item IDs in the catalog.

    Returns:
        Gini Index as a float in [0, 1]. Higher values means more bias.
    """

    catalog_set = set(catalog)
    # Flatten across all users
    flattened_recs = recommendations.flatten().tolist()
    unknown = set(flattened_recs) - catalog_set
    if unknown:
        raise ValueError(f"Recommendations contain items not in catalog: {unknown}")

    n = len(catalog)
    # and count per-item recommendation frequency
    counts = Counter(flattened_recs)
    freq = sorted(counts.get(item, 0) for item in catalog)  # ascending order

    total = sum(freq)
    if total == 0:
        return 0.0

    weighted_sum = sum((i + 1) * x for i, x in enumerate(freq))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def catalog_coverage(rec, catalog):
    """
    Calculate the catalog coverage of recommendations.
    Args:
        rec (torch.Tensor): Recommendations tensor of shape (n_users, k) containing item indices.
        catalog (torch.Tensor): Tensor of all available candidate items.
    Returns:
        float: Ratio of unique recommended items to total catalog.
    """

    return rec.unique().shape[0] / len(catalog)
