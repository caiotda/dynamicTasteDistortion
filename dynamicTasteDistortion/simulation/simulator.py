import torch

import numpy as np
import pandas as pd

from calibratedRecs.calibratedRecs.calibrationUtils import (
    build_item_genre_distribution_tensor,
    preprocess_dataframe_for_calibration,
)
from calibratedRecs.metrics import mace
from dynamicTasteDistortion.simulation.simulationUtils import (
    random_rec,
    get_feedback_for_predictions,
)
from dynamicTasteDistortion.simulationConstants import (
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
        user_sample=None,
        bootstrapping_rounds=10,
        bootstrapped_df=None,
    ):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.initial_date is None:
            self.initial_date = pd.Timestamp.now().timestamp()

        self.users = torch.tensor(users, device=self.device)

        self.items = torch.tensor(
            list(oracle_matrix[ITEM_COL].drop_duplicates()), device=self.device
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

        ratings_df = preprocess_dataframe_for_calibration(self.oracle_matrix)
        n_items = ratings_df[ITEM_COL].max() + 1
        n_users = self.ratings_df[USER_COL].max() + 1
        self.item_distribution_tensor = build_item_genre_distribution_tensor(
            ratings_df, n_items
        )

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
            rec = random_rec(self.items, n_users, k)
        else:
            rec = self.model.predict(
                user=self.users, k=k, candidates=self.items, mask=mask
            )[0]

        feedback_matrix = get_feedback_for_predictions(self.oracle_matrix, rec)
        indices = get_matrix_coordinates(feedback_matrix)

        users_indices, click_positions = indices[:, 0].tolist(), indices[:, 1].tolist()
        user_ids = [self.user_idx_to_id[idx] for idx in users_indices]

        feedbacks = feedback_matrix.flatten().tolist()
        items = rec.flatten().tolist()

        timestamps = [
            (self.timestamp_distribution[user].rvs(1)[0] / 60) + self.initial_date
            for user in user_ids
        ]
        entries = list(
            zip(users_indices, items, feedbacks, click_positions, timestamps)
        )
        interaction_df = pd.DataFrame(
            entries, columns=["user", "item", "relevant", "clicked_at", "timestamp"]
        )

        interaction_df.loc[interaction_df["relevant"] != 1.0, "clicked_at"] = np.nan
        interaction_df.loc[interaction_df["relevant"] != 1.0, "timestamp"] = np.nan

        return interaction_df

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
            [], columns=["user", "item", "relevant", "clicked_at", "timestamp"]
        )
        for _ in range(bootstrapping_rounds):
            round_df = self.simulate_user_feedback(
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

        item2genreMap = (
            self.oracle_matrix[[ITEM_COL, GENRES_COL]]
            .set_index(ITEM_COL)[GENRES_COL]
            .to_dict()
        )
        user2history = (
            self.click_matrix.groupby(USER_COL)
            .agg({ITEM_COL: list})
            .to_dict()[ITEM_COL]
        )

        boostrapped_df = self.click_matrix.copy()
        maces = []
        for round_idx in tqdm(range(rounds), desc="Processing rounds..."):
            round_df = self.simulate_user_feedback(
                mask=None, feedback_from_bootstrap=True, k=k
            )
            if round_idx % L == 0:
                print("retraining model...")
                # TODO double check
                _ = self.model.fit(boostrapped_df)
                print("Calculating mace")
                rec_df_grouped = (
                    boostrapped_df.groupby(USER_COL)
                    .agg({ITEM_COL: list})
                    .reset_index()
                    .rename(columns={ITEM_COL: "rec"})
                )
                iteration_mace = mace(
                    rec_df=rec_df_grouped,
                    user2history=user2history,
                    recCol="rec",
                    item2genreMap=item2genreMap,
                )
                maces.append(iteration_mace)
            if round_idx % 100 == 0:
                boostrapped_df.to_csv(
                    f"data/movielens/no_calibration_sim_up_to_round_{round_idx}"
                )
            boostrapped_df = pd.concat([boostrapped_df, round_df], ignore_index=True)

        return boostrapped_df, maces
