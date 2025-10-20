import math
import random


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



def get_user_feedback_for_item(user, item ,k, oraclePreferenceMatrix, ratingDeltaDistribution):
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
        timestamp = ratingDeltaDistribution[user].rvs(1)[0] / 60
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
    for idx, item in enumerate(rec_list):
        user, item, feedback, clicked_at, timestamp  = get_user_feedback_for_item(user, item, idx+1, matrix, ratingDeltaDistribution, initial_time)
        if (timestamp is not None):
            initial_time = timestamp
        feedback = (user, item, feedback, clicked_at, timestamp)
        results.append(feedback)
    return results, initial_time