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


def catalog_coverage(rec, candidates):
    """
    Calculate the catalog coverage of recommendations.
    Args:
        rec (torch.Tensor): Recommendations tensor of shape (n_users, k) containing item indices.
        candidates (torch.Tensor): Tensor of all available candidate items.
    Returns:
        float: Ratio of unique recommended items to total candidates.
    """

    return rec.unique().shape[0] / len(candidates)