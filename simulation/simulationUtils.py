import math
import random
import torch

from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_user_preference_for_item(user, item, matrix):
    # Returns the preference of a user for an item according to an oracle preference matrix
    user_ratings = matrix[matrix["user"] == user]
    return user_ratings[user_ratings["item"] == item].rating.item()

def click_model(k):
    """
        Simulates a click model for a ranked list position.

        Args:
            k (int): The rank position (1-based index) of the item.

        Returns:
            bool: True if the item is clicked (examined), False otherwise.

        Notes:
            The probability of examination is determined by a logarithmic decay function,
            where higher-ranked items (lower k) have a higher chance of being examined.
    """
    lambda_k = 1/math.log(k+1,2)
    examination_probability = random.random()
    if examination_probability <= lambda_k:
        return True
    return False



def get_user_feedback_for_item(user, item ,k, oraclePreferenceMatrix, ratingDeltaDistribution, initial_time):
    """
    Simulates user feedback for a given item at position k in the recommendation list.
    ins:
        user - user id
        item - item id
        k - position in the recommendation list (1-based index)
        oraclePreferenceMatrix - dataframe with user-item preferences
        ratingDeltaDistribution - dictionary mapping user ids to their corresponding
                                  exponential distribution for time between interactions
        initial_time - the initial timestamp to calculate the interaction time
    outs:
        user - user id
        item - item id     
        feedback - user feedback (1 if clicked, 0 if not clicked, None if ignored)
        clicked_at - position in the recommendation list where the item was clicked (1-based index)
        timestamp - timestamp of the interaction
    """
    preference = get_user_preference_for_item(user, item, oraclePreferenceMatrix)
    observed = click_model(k)
    relevant = bool(preference)
    should_click = observed and relevant
    if (should_click):
        feedback = 1
        clicked_at = k
        # Sample one delta_t random variable following the users
        # Exponential time distribution
        timestamp = (ratingDeltaDistribution[user].rvs(1)[0] / 60) + initial_time
    else:
        if (observed):
            # Case where an item was observed, but isn´t relevant -> negative example for BPR
            feedback = 0
        else:
            # Case where an item was neither observed or relevant -> we will ignore this training instance in this loop
            feedback = None
        clicked_at = None
        timestamp = None
        
    # If user clicked the item, record the position it was in
    # feedback = 1 if user examined and clicked, 0 if user examined and not clicked,
    # None if otherwise
    return user, item, feedback, clicked_at, timestamp


def map_recommendation_to_feedback(user, rec_list, matrix, initial_time, ratingDeltaDistribution):
    """
    Maps a list of recommendations to user feedback signals.
    ins:
        user - user id
        rec_list - list of recommended items
        matrix - dataframe with user-item preferences
        user_to_up_to_date_timestamp - dataframe mapping users to their last interaction timestamp
        ratingDeltaDistribution - dictionary mapping user ids to their corresponding
                                  exponential distribution for time between interactions
    outs:
        results - list of tuples (user, item, feedback, clicked_at, timestamp)
        final_time - the last recorded timestamp of interaction after processing all recommendations
    """
    
    results = []
    final_time = initial_time
    for idx, item in enumerate(rec_list):
        user, item, feedback, clicked_at, timestamp  = get_user_feedback_for_item(user, item, idx+1, matrix, ratingDeltaDistribution, initial_time)
        if (timestamp is not None):
            initial_time = timestamp
            if(timestamp > final_time):
                final_time = timestamp
        feedback = (user, item, feedback, clicked_at, timestamp)
        results.append(feedback)
    return results, final_time

def get_candidate_items(user, D, unique_items):
    user_history = set(D[D["user"] == user]["item"])
    candidate_items = [item for item in unique_items if item not in user_history]
    return candidate_items


def random_rec(candidates, k):
    return random.sample(candidates, k)

def simulate_user_feedback(user, candidate_items, preference_matrix, k, user_to_up_to_date_timestamp, userToExpDistribution, recommend):
    """
        Simulates user feedback for a given user by recommending k items and mapping the recommendations to feedback
        ins:
            user - user id
            candidate_items - list of candidate items
            preference_matrix - Oracle preference matrix dataframe
            k - number of items to recommend
            user_to_up_to_date_timestamp - dataframe mapping users to their last interaction timestamp
            userToExpDistribution - dictionary mapping user ids to their corresponding
                                        exponential distribution for time between interactions
            recommend - A recommendation function that necessarily receives the user, the cutoff k, and a list
            of candidates.
        outs:
            row - list of tuples (user, item, feedback, clicked_at, timestamp)
            user_to_up_to_date_timestamp - updated dataframe mapping users to their last interaction timestamp
    """
    rec = recommend(user=torch.tensor(user, device=device), k=k, candidates=candidate_items)
    # Generates a user feedback to each recommendation and the last recorded time of interaction in this recommendation
    row, last_time = map_recommendation_to_feedback(user, rec, preference_matrix, user_to_up_to_date_timestamp, userToExpDistribution)
    user_to_up_to_date_timestamp.loc[user_to_up_to_date_timestamp["user"] == user, "delta_from_start"] = last_time
    return row, user_to_up_to_date_timestamp