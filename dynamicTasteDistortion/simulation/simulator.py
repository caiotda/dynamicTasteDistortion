import torch
import os
import copy

import numpy as np
import pandas as pd


from calibratedRecs.constants import UNKNOWN_GENRE
from calibratedRecs.calibrationUtils import (
    build_item_genre_distribution_tensor,
    preprocess_dataframe_for_calibration,
    build_user_genre_history_distribution,
)

from calibratedRecs.reranking_utils import rerank_by_calibration

from calibratedRecs.metrics import mace, get_avg_kl_div
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
        base_artifacts_path=None,
        bootstrapping_rounds=10,
        bootstrapped_df=None,
        ignore_oracle_matrix=False,
        calibration_type=None,
    ):
        device = (
            model.device
            if model is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.top_k_for_evaluation = 10
        self.calibration_type = calibration_type
        self.timestamp_distribution = user_timestamp_distribution
        # TODO: isso precisa ser meio refatorado.
        self.ignore_oracle_matrix = ignore_oracle_matrix
        self.user_idx_to_id = {
            idx: user_id
            for idx, user_id in enumerate(self.timestamp_distribution.keys())
        }

        users = list(self.user_idx_to_id.values())
        self.oracle_matrix = (
            oracle_matrix[oracle_matrix[USER_COL].isin(users)]
            if oracle_matrix is not None
            else None
        )
        self.use_random_rec = True if model is None else False
        self.model = model
        self.initial_date = initial_date

        if self.initial_date is None:
            self.initial_date = pd.Timestamp.now().timestamp()

        self.users = torch.tensor(users, device=device)

        self.items = torch.tensor(
            list(oracle_matrix[ITEM_COL].drop_duplicates()), device=device
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
        self.n_items = ratings_df[ITEM_COL].max() + 1
        self.n_users = ratings_df[USER_COL].max() + 1
        self.p_g_i = build_item_genre_distribution_tensor(ratings_df, self.n_items)

        self.base_artifacts_path = base_artifacts_path
        if base_artifacts_path is not None and not os.path.exists(
            self.base_artifacts_path
        ):
            os.makedirs(self.base_artifacts_path)

    def simulate_user_feedback(self, rec, score):
        """
        Simulates user feedback for a batch of users by recommending k items and mapping the recommendations to feedback.

        Args:
            rec (torch.Tensor): Tensor containing the ordered recommendation.
            score (torch.Tensor): Tensor containing the ordered score of each item in the recommendation.

        Returns:
            pd.DataFrame: A dataframe containing the simulated interactions with the following schema:
                - user: User indices.
                - item: Recommended item indices.
                - click: Feedback values (1 for positive, 0 for negative, NaN for no interaction).
                - clicked_at: Click positions in the recommendation list (NaN if no click occurred).
                - timestamp: Interaction timestamps (NaN if no interaction occurred).
        """

        if self.ignore_oracle_matrix:
            feedback_matrix = get_feedback_for_predictions(None, rec)
        else:
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

        n_users = self.users.max() + 1
        rec, score = random_rec(self.items, n_users, k)

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
            round_df = self.simulate_user_feedback(rec=rec, score=score, k=k)
            bootstrapped_df = pd.concat([bootstrapped_df, round_df], ignore_index=True)

        bootstrapped_df["relevant"] = bootstrapped_df["relevant"].fillna(0).astype(int)
        bootstrapped_df["clicked_at"] = (
            bootstrapped_df["clicked_at"].fillna(-1).astype(int)
        )
        return bootstrapped_df

    def _recommend(self, users_history, k, mask=None):

        if self.use_random_rec:
            n_users = self.users.max() + 1
            rec, score = random_rec(self.items, n_users, k)
        else:
            rec, score = self.model.recommend(
                users=self.users, k=k, candidates=self.items, mask=mask
            )

        if self.calibration_type is not None:
            rec, score = rerank_by_calibration(
                recs=rec,
                scores=score,
                ratings_df=users_history,
                n_users=self.n_users,
                n_items=self.n_items,
                calib_k=self.top_k_for_evaluation,
                item2genreMap=self.item2genreMap,
            )
        return rec, score

    def _mask_previously_seen_items(self, users_history):
        mask = torch.ones((self.n_users, self.n_items), dtype=torch.float32)
        seen = users_history[users_history["relevant"] == 1.0][[USER_COL, ITEM_COL]]
        user_idx = torch.tensor(seen[USER_COL].astype(int).values, dtype=torch.long)
        item_idx = torch.tensor(seen[ITEM_COL].astype(int).values, dtype=torch.long)

        mask[user_idx, item_idx] = -1.0
        return mask

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
        H_0 = boostrapped_df
        H_0["genres"] = (
            H_0["item"]
            .map(self.item2genreMap)
            .apply(lambda x: x if isinstance(x, list) else [UNKNOWN_GENRE])
        )
        maces = []
        kl_divs = []
        initial_model = copy.deepcopy(self.model)

        user_history_tensor = build_user_genre_history_distribution(
            H_0,
            self.p_g_i,
            n_users=self.n_users,
            n_items=self.n_items,
            weight_col="constant",
        )

        for round_idx in tqdm(range(1, rounds + 1), desc="Processing rounds..."):
            mask = self._mask_previously_seen_items(boostrapped_df)
            rec, score = self._recommend(users_history=H_0, k=k, mask=mask)

            round_df = self.simulate_user_feedback(
                rec=rec,
                score=score,
            )

            rec_genre_distribution_tensor = build_user_genre_history_distribution(
                round_df,
                self.p_g_i,
                n_users=self.n_users,
                n_items=self.n_items,
                weight_col="rating",
            )
            iteration_mace = mace(
                rec_df=round_df,
                p_g_u=user_history_tensor,
                p_g_i=self.p_g_i,
                k=self.top_k_for_evaluation,
            )

            iteration_avg_kl_div = get_avg_kl_div(
                self.users, user_history_tensor, rec_genre_distribution_tensor
            )
            kl_divs.append(iteration_avg_kl_div)
            maces.append(iteration_mace)

            boostrapped_df = pd.concat([boostrapped_df, round_df], ignore_index=True)

            if round_idx % L == 0:
                if not self.use_random_rec:
                    print("retraining model...")
                    self.model = copy.deepcopy(initial_model)
                    self.model.to(initial_model.device)
                    _ = self.model.fit(boostrapped_df, debug=False)
                boostrapped_df = round_df

        return boostrapped_df, maces, kl_divs
