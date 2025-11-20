import math
import random
import torch
import pickle

from tqdm import tqdm
from scipy.stats import expon

import pandas as pd

import sys
import os
from calibratedRecs.metrics import mace
from tasteDistortionOnDynamicRecs.simulation.simulationUtils import get_candidate_items, simulate_user_feedback
from tasteDistortionOnDynamicRecs.simulationConstants import USER_COL, ITEM_COL, GENRES_COL

class Simulator:
    def __init__(self, oracle_matrix, model, num_rounds, initial_date, user_sample=None):
        self.timestamp_distribution = self.setup_user_timestamp_distribution()
        self.user_idx_to_id = {idx: user_id for idx, user_id in enumerate(self.timestamp_distribution.keys())}

        if user_sample is None:
            users = list(self.user_idx_to_id.values())
        else:
            users = user_sample
        self.oracle_matrix = oracle_matrix[oracle_matrix[USER_COL].isin(users)]
        self.model = model
        self.rounds = num_rounds
        self.initial_date = initial_date

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.initial_date is None:
            self.initial_date = pd.Timestamp.now().timestamp()



        self.users = torch.tensor(users , device=self.device)

        self.items = torch.tensor(list(oracle_matrix[ITEM_COL].drop_duplicates()), device=self.device)

        self.click_matrix = self.bootstrap_clicks()

    def setup_user_timestamp_distribution(self):

        user_to_time_delta = pd.read_csv("../data/movielens-1m/median_time_diff_per_user.csv").set_index("userId")

        user_to_exp_distribution = {
            user: expon(scale=row["median_timestamp_diff"])
            for user, row in user_to_time_delta.iterrows()
        }

        
        return user_to_exp_distribution


    def simulate_round_of_recommendation(self, recommendation_function):
        rows_to_append = []
        for user in self.users:
            user_to_up_to_date_timestamp = self.timestamp_distribution.copy()
            initial_time = user_to_up_to_date_timestamp.loc[user_to_up_to_date_timestamp["user"] == user, "delta_from_start"].squeeze()
            candidate_items = get_candidate_items(user, self.click_matrix, self.items)
            user_to_up_to_date_timestamp = self.user_to_up_to_date_timestamp
            row, user_to_up_to_date_timestamp = simulate_user_feedback(
                user,
                candidate_items,
                self.oracle_matrix,
                self.k,
                initial_time,
                user_to_up_to_date_timestamp,
                recommend=recommendation_function
            )
            rows_to_append.extend(row)
        self.timestamp_distribution = user_to_up_to_date_timestamp
        round_df = pd.DataFrame(rows_to_append, columns=self.click_df.columns)
        return round_df

    def bootstrap_clicks(self, k=20):
        """
        Given unique users and unique items, recommend up to k items to every user
        using a preference matrix as a relevancy model and using a click model
        to simulate probability of user examinating an item.

        Feedback signal will be fed to the D matrix.

        In order to ensure enough feedback data to train a model, we run the boostrap process for a total of an arbitrary number
        of rounds, using the recommend function to generate recommendations and simulating the feedbacks.
        
        """
        
        boostrapped_df = pd.DataFrame([], columns=["user", "item", "feedback", "clicked_at", "timestamp"])
        for _ in range(self.rounds):
            round_df = simulate_user_feedback(
                users=self.users,
                candidate_items=self.items,
                mask=None,
                oracle_matrix=self.oracle_matrix,
                rating_delta_distribution=self.timestamp_distribution,
                model=None,
                feedback_from_bootstrap=True,
                user_idx_to_id_map=self.user_idx_to_id,
                k=k
            )
            boostrapped_df = pd.concat([boostrapped_df, round_df], ignore_index=True)

        return boostrapped_df   



    def simulate(
        self,
        click_matrix,
        rounds,
        recommend,
        L=10, 
    ):
        """
        Simulates a dynamic recommendation setting.

        Parameters
        ----------
        D : pd.DataFrame
            Initial feedback data containing user-item interactions.
        model : torch.nn.Module
            Recommendation model with a predict_flat method.
        unique_users : list
            List of unique user IDs to simulate.
        unique_items : list
            List of unique item IDs to recommend.
        oracleMatrix : pd.DataFrame
            Matrix containing oracle user preferences for items.
        userToExpDistribution : dict
            Mapping from user ID to their timestamp distribution.
        item2genreMap : dict
            Mapping from item ID to its genres.
        k : int, optional
            Number of items to recommend per user per round (default=100).
        rounds : int, optional
            Number of simulation rounds (default=1000).
        L : int, optional
            Retrain model every L rounds (default=10).
        initial_date : float, optional
            Initial timestamp for simulation (default=None).

        Returns
        -------
        final_df : pd.DataFrame
            DataFrame containing all simulated feedback.
        maces : list
            List of MACE metric values computed every L rounds.
        """

        item2genreMap = (
            self.oracle_matrix[[ITEM_COL, GENRES_COL]]
            .set_index(ITEM_COL)[GENRES_COL]
            .to_dict()
        )
        user2history = click_matrix.groupby(USER_COL).agg({ITEM_COL: list}).to_dict()[ITEM_COL]

        new_df = click_matrix.copy()
        maces = []
        for round in tqdm(range(1, rounds + 1), desc="Rounds"):
            recommendation_df = self.simulate_round_of_recommendation(recommend)
            if (round % L == 0):
                print("retraining model...")
                # TODO que porra é essa mano
                model = train(model, new_df)
                print("Calculating mace")
                rec_df_grouped = recommendation_df.groupby(USER_COL).agg({ITEM_COL: list}).reset_index().rename(columns={ITEM_COL: "rec"})
                iteration_mace = mace(df=rec_df_grouped, user2history=user2history, recCol='rec', item2genreMap=item2genreMap)
                maces.append(iteration_mace)
            if (round % 100 == 0):
                recommendation_df.to_csv(f"data/movielens/no_calibration_sim_up_to_round_{round}")
            new_df = pd.concat([new_df, recommendation_df], ignore_index=True)
        final_df = pd.concat([click_matrix, new_df])
        final_df.loc[final_df["timestamp"].notnull(), "timestamp"] += self.initial_date
        return final_df, maces