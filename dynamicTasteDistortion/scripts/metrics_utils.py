from typing import Counter

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
