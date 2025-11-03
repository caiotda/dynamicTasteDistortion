import math
import random
import torch

from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_user_preference_for_item(user, item, matrix):
    user_ratings = matrix[matrix["user"] == user]
    return user_ratings[user_ratings["item"] == item].rating.item()

def click_model(k):
    lambda_k = 1/math.log(k+1,2)
    examination_probability = random.random()
    if examination_probability <= lambda_k:
        return True
    return False


def get_inverse_propensity_click_score(position):
    # Given a click position, this funtion returns the invense propensity, 
    # usefull to debias the data later.
    return - 1/math.log(position+1,2)



def get_user_feedback_for_item(user, item ,k, oraclePreferenceMatrix, ratingDeltaDistribution, initial_time):
    # Build a mapping from user to their timestamp distribution
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


def map_recommendation_to_feedback(user, rec_list, matrix, user_to_up_to_date_timestamp, ratingDeltaDistribution):
    initial_time = user_to_up_to_date_timestamp.loc[user_to_up_to_date_timestamp["user"] == user, "delta_from_start"].squeeze()
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
    rec = recommend(user=torch.tensor(user, device=device), k=k, candidates=candidate_items)
    # Generates a user feedback to each recommendation and the last recorded time of interaction in this recommendation
    row, last_time = map_recommendation_to_feedback(user, rec, preference_matrix, user_to_up_to_date_timestamp, userToExpDistribution)
    user_to_up_to_date_timestamp.loc[user_to_up_to_date_timestamp["user"] == user, "delta_from_start"] = last_time
    return row, user_to_up_to_date_timestamp


def bootstrap_clicks(D, unique_users, unique_items, preference_matrix, userToExpDistribution, k=20, rounds=10, initial_date=None):
    """
    Given unique users and unique items, recommend up to k items to every user
    using a preference matrix as a relevancy model and using a click model
    to simulate probability of user examinating an item.

    Feedback signal will be fed to the D matrix.

    We run the boostrap process for a total of an arbitrary number of rounds,
    in order to ensure enough feedback data to train a model.
    """


    if initial_date is None:
        initial_date = pd.Timestamp.now().timestamp()

    # Setup a dataframe which maps each user to the time delta from its first interaction.
    user_to_up_to_date_timestamp = pd.DataFrame({
        "user": unique_users,
        "delta_from_start": 0.0
    })
    
    # Maps each user to its corresponding exponential distribution, which models the average time between interactions
    # for each user
    user_to_up_to_date_timestamp["timestamp_dist"] = user_to_up_to_date_timestamp["user"].map(userToExpDistribution)
    new_df = D.copy()
    for round in range(rounds):
        rows_to_append = []
        for user in tqdm(unique_users, desc=f"Processing users (round {round+1}/{rounds})..."):
            candidate_items = get_candidate_items(user, D, unique_items)
            row, user_to_up_to_date_timestamp = simulate_user_feedback(
                user,
                candidate_items,
                preference_matrix,
                k,
                user_to_up_to_date_timestamp,
                userToExpDistribution,
                recommend=random_rec
            )
            #row, user_to_up_to_date_timestamp = map_recommendation_to_feedback(user, recs, preference_matrix, user_to_up_to_date_timestamp)
            rows_to_append.extend(row)
        round_df = pd.DataFrame(rows_to_append, columns=new_df.columns)
        new_df = pd.concat([new_df, round_df], ignore_index=True)
    final_df = pd.concat([D, new_df])
    # Offsets the recorded timestamps by initial_date
    final_df.loc[final_df["timestamp"].notnull(), "timestamp"] += initial_date
    return final_df