import torch
import os

import numpy as np
import pandas as pd


from calibratedRecs.calibration import Calibration
from calibratedRecs.calibrationUtils import (
    build_item_genre_distribution_tensor,
    preprocess_dataframe_for_calibration,
    build_user_genre_history_distribution,
)
from calibratedRecs.metrics import mace, get_avg_kl_div
from dynamicTasteDistortion.simulation.simulationUtils import (
    random_rec,
    get_feedback_for_predictions,
    rerank_with_calib,
)
from dynamicTasteDistortion.simulationConstants import (
    RESULTS_PATH,
    USER_COL,
    ITEM_COL,
    GENRES_COL,
)
from dynamicTasteDistortion.simulation.tensorUtils import (
    get_matrix_coordinates,
)


from tqdm import tqdm


class Simulator:
    def __init__(
        self,
        oracle_matrix,
        model,
        initial_date,
        user_timestamp_distribution,
        base_artifacts_path,
        user_sample=None,
        bootstrapping_rounds=10,
        bootstrapped_df=None,
        should_calibrate=False,
    ):

        self.should_calibrate = should_calibrate
        self.timestamp_distribution = user_timestamp_distribution
        self.user_idx_to_id = {
            idx: user_id
            for idx, user_id in enumerate(self.timestamp_distribution.keys())
        }

        if user_sample is None:
            users = list(self.user_idx_to_id.values())
        else:
            users = user_sample
        self.oracle_matrix = oracle_matrix[oracle_matrix[USER_COL].isin(users)]
        self.model = model
        self.initial_date = initial_date

        if self.initial_date is None:
            self.initial_date = pd.Timestamp.now().timestamp()

        self.users = torch.tensor(users, device=model.device)

        self.items = torch.tensor(
            list(oracle_matrix[ITEM_COL].drop_duplicates()), device=model.device
        )

        if (bootstrapped_df is not None) and (not bootstrapped_df.empty):
            self.click_matrix = bootstrapped_df
        else:
            self.click_matrix = self.bootstrap_clicks(
                k=100, bootstrapping_rounds=bootstrapping_rounds
            )

        self.item2genreMap = (
            self.oracle_matrix[[ITEM_COL, GENRES_COL]]
            .set_index(ITEM_COL)[GENRES_COL]
            .to_dict()
        )

        self.click_matrix[GENRES_COL] = self.click_matrix[ITEM_COL].map(
            self.item2genreMap
        )

        ratings_df = preprocess_dataframe_for_calibration(self.oracle_matrix)
        self.n_items = ratings_df[ITEM_COL].max() + 1
        self.n_users = ratings_df[USER_COL].max() + 1
        self.p_g_i = build_item_genre_distribution_tensor(ratings_df, self.n_items)

        self.base_artifacts_path = base_artifacts_path
        if not os.path.exists(self.base_artifacts_path):
            os.makedirs(self.base_artifacts_path)

    def simulate_user_feedback(self, mask, k, feedback_from_bootstrap=False):
        """
        Simulates user feedback for a batch of users by recommending k items and mapping the recommendations to feedback.

        Args:
            mask (torch.Tensor): 2D tensor indicating items to ignore during recommendation (e.g., previously interacted items).
            k (int): Number of items to recommend per user.
            feedback_from_bootstrap (bool, optional): If True, generates random recommendations instead of using the model. Defaults to False.

        Returns:
            pd.DataFrame: A dataframe containing the simulated interactions with the following schema:
                - user: User indices.
                - item: Recommended item indices.
                - click: Feedback values (1 for positive, 0 for negative, NaN for no interaction).
                - clicked_at: Click positions in the recommendation list (NaN if no click occurred).
                - timestamp: Interaction timestamps (NaN if no interaction occurred).
        """
        if feedback_from_bootstrap:
            n_users = self.users.max() + 1
            rec, score = random_rec(self.items, n_users, k)
        else:
            rec, score = self.model.recommend(
                users=self.users, k=k, candidates=self.items, mask=mask
            )

            if self.should_calibrate:
                calibration_params = {
                    "weight": "linear_time",
                    "distribution_mode": "steck",
                    "lambda": 0.99,
                }
                rec, score = rerank_with_calib(
                    click_df=self.click_matrix,
                    users=self.users,
                    rec=rec,
                    scores=score,
                    calibration_params=calibration_params,
                )

        feedback_matrix = get_feedback_for_predictions(self.oracle_matrix, rec)
        indices = get_matrix_coordinates(feedback_matrix)

        users_indices, click_positions = indices[:, 0].tolist(), indices[:, 1].tolist()
        user_ids = [self.user_idx_to_id[idx] for idx in users_indices]

        feedbacks = feedback_matrix.flatten().tolist()
        items = rec.flatten().tolist()
        scores = score.flatten().tolist()
        constant = [1.0] * len(scores)

        timestamps = [
            (self.timestamp_distribution[user].rvs(1)[0] / 60) + self.initial_date
            for user in user_ids
        ]
        entries = list(
            zip(
                users_indices,
                items,
                feedbacks,
                click_positions,
                timestamps,
                scores,
                constant,
            )
        )
        interaction_df = pd.DataFrame(
            entries,
            columns=[
                "user",
                "item",
                "relevant",
                "clicked_at",
                "timestamp",
                "rating",
                "constant",
            ],
        )

        interaction_df.loc[interaction_df["relevant"] != 1.0, "clicked_at"] = np.nan
        interaction_df.loc[interaction_df["relevant"] != 1.0, "timestamp"] = np.nan
        interaction_df["constant"] = 1.0  # For calibration purposes
        return interaction_df, rec

    def bootstrap_clicks(self, k=20, bootstrapping_rounds=5):
        """
        Given unique users and unique items, recommend up to k items to every user
        using a preference matrix as a relevancy model and using a click model
        to simulate probability of user examinating an item.

        Feedback signal will be fed to the D matrix.

        In order to ensure enough feedback data to train a model, we run the boostrap process for a total of an arbitrary number
        of rounds, using the recommend function to generate recommendations and simulating the feedbacks.

        """

        bootstrapped_df = pd.DataFrame(
            [],
            columns=[
                "user",
                "item",
                "relevant",
                "clicked_at",
                "timestamp",
                "rating",
                "constant",
            ],
        )
        for _ in range(bootstrapping_rounds):
            round_df, _ = self.simulate_user_feedback(
                mask=None, feedback_from_bootstrap=True, k=k
            )
            bootstrapped_df = pd.concat([bootstrapped_df, round_df], ignore_index=True)

        bootstrapped_df["relevant"] = bootstrapped_df["relevant"].fillna(0).astype(int)
        bootstrapped_df["clicked_at"] = (
            bootstrapped_df["clicked_at"].fillna(-1).astype(int)
        )
        return bootstrapped_df

    def simulate(self, k=100, L=10, rounds=10_000):
        """
        Simulates a dynamic recommendation setting.

        Parameters
        ----------
        k : int, optional
            Number of items to recommend per user per round (default=100).
        L : int, optional
            Retrain the model every L rounds (default=10).
        rounds : int, optional
            Total number of simulation rounds (default=10,000).

        Returns
        -------
        final_df : pd.DataFrame
            DataFrame containing all simulated feedback, including user-item interactions, clicks, and timestamps.
        maces : list
            List of MACE metric values computed every L rounds to evaluate recommendation quality.
        """

        boostrapped_df = self.click_matrix.copy()
        boostrapped_df["constant"] = 1.0
        maces = []
        kl_divs = []
        for round_idx in tqdm(range(1, rounds + 1), desc="Processing rounds..."):
            round_df, round_rec = self.simulate_user_feedback(
                mask=None, feedback_from_bootstrap=False, k=k
            )
            if round_idx % L == 0:
                print("retraining model...")
                _ = self.model.fit(boostrapped_df)
                print("Calculating mace")
                user_history_tensor = build_user_genre_history_distribution(
                    boostrapped_df,
                    self.p_g_i,
                    n_users=self.n_users,
                    n_items=self.n_items,
                    weight_col="constant",
                )

                rec_tensor = build_user_genre_history_distribution(
                    round_df,
                    self.p_g_i,
                    n_users=self.n_users,
                    n_items=self.n_items,
                    weight_col="rating",  # Prediction
                )

                iteration_mace = mace(
                    rec_df=round_df.groupby(USER_COL).agg(list).reset_index(),
                    p_g_u=user_history_tensor,
                    p_g_i=self.p_g_i,
                )
                iteration_avg_kl_div = get_avg_kl_div(
                    self.users, user_history_tensor, rec_tensor
                )
                kl_divs.append(iteration_avg_kl_div)
                maces.append(iteration_mace)
            if round_idx % 100 == 0:
                boostrapped_df.to_csv(
                    f"{self.base_artifacts_path}/simulated_recommendation_round_{round_idx}.csv"
                )
            boostrapped_df = pd.concat([boostrapped_df, round_df], ignore_index=True)

        return boostrapped_df, maces, kl_divs
