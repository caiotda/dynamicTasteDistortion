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

from calibratedRecs.weight_functions import get_linear_time_weight_rating
from calibratedRecs.reranking_utils import rerank_by_calibration
from calibratedRecs.mappings import CALIBRATION_MODE_TO_COL_NAME
from calibratedRecs.metrics import mace, get_avg_kl_div
from dynamicTasteDistortion.simulation.simulationUtils import (
    build_examination_matrix,
    get_feedback_matrix,
    map_prediction_to_preferences,
    random_rec,
    update_preference_matrix,
)
from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
    GENRES_COL,
    TIMESTAMP_COL,
    RATING_COL,
)
from dynamicTasteDistortion.simulation.tensorUtils import (
    binary_to_bipolar,
    get_matrix_coordinates,
    pandas_df_to_sparse_tensor,
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
        num_interactions_bootstrapped=1_000_000,
        bootstrapped_df=None,
        ignore_oracle_matrix=False,
        calibration_type=None,
        preference_update_rate=0,
    ):
        self.device = (
            model.device
            if model is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.preference_update_rate = preference_update_rate
        self.top_k_for_evaluation = 10
        self.calibration_type = calibration_type
        self.timestamp_distribution = user_timestamp_distribution
        self.ignore_oracle_matrix = ignore_oracle_matrix
        self.user_idx_to_id = {
            idx: user_id
            for idx, user_id in enumerate(self.timestamp_distribution.keys())
        }

        users = list(self.user_idx_to_id.values())
        filtered_oracle_matrix = (
            oracle_matrix[oracle_matrix[USER_COL].isin(users)]
            if oracle_matrix is not None
            else None
        )
        self.n_users = filtered_oracle_matrix[USER_COL].max() + 1
        self.n_items = filtered_oracle_matrix[ITEM_COL].max() + 1
        self.item2genreMap = (
            filtered_oracle_matrix[[ITEM_COL, GENRES_COL]]
            .set_index(ITEM_COL)[GENRES_COL]
            .to_dict()
        )
        self.oracle_tensor = pandas_df_to_sparse_tensor(filtered_oracle_matrix)

        self.use_random_rec = True if model is None else False
        self.model = model
        self.initial_date = initial_date

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
                k=100, num_interactions_bootstrapped=num_interactions_bootstrapped
            )
        self.click_matrix[GENRES_COL] = (
            self.click_matrix[ITEM_COL]
            .map(self.item2genreMap)
            .apply(
                lambda x: tuple(x) if isinstance(x, list) else tuple([UNKNOWN_GENRE])
            )
        )
        self.click_matrix = preprocess_dataframe_for_calibration(self.click_matrix)
        self.p_g_i = build_item_genre_distribution_tensor(
            self.click_matrix, self.n_items
        )

        self.base_artifacts_path = base_artifacts_path
        if base_artifacts_path is not None and not os.path.exists(
            self.base_artifacts_path
        ):
            os.makedirs(self.base_artifacts_path)

    def get_feedback_for_predictions(self, predictions):
        # Simulates feedback only through click position.
        if self.ignore_oracle_matrix:
            preferences_matrix = torch.ones_like(
                predictions, dtype=torch.int8, device=self.device
            )
            should_update_preferences = False
        else:
            should_update_preferences = True
            preferences_matrix = map_prediction_to_preferences(
                self.oracle_tensor, predictions
            )

        examination_matrix = build_examination_matrix(
            predictions, shape=(self.n_users, self.n_items)
        )
        feedback_matrix = get_feedback_matrix(predictions, preferences_matrix)

        mapped_feedback = torch.where(
            feedback_matrix == 0,
            torch.tensor(float("nan"), device=feedback_matrix.device),
            torch.where(
                feedback_matrix < 0,
                torch.tensor(0, device=feedback_matrix.device),
                torch.tensor(1, device=feedback_matrix.device),
            ),
        )
        # Update user preferences after examining recommendations
        self.oracle_tensor = (
            update_preference_matrix(
                preference_matrix=self.oracle_tensor,
                examination_matrix=examination_matrix,
                preference_update_rate=self.preference_update_rate,
            )
            if should_update_preferences
            else self.oracle_tensor
        )

        return mapped_feedback

    def simulate_user_feedback(self, rec, score):
        """
        Simulates user feedback for a batch of recommendations.

        Args:
            rec (torch.Tensor): Ordered recommendations, shape (n_users, k).
            score (torch.Tensor): Corresponding scores, shape (n_users, k).

        Returns:
            interaction_df (pd.DataFrame): Simulated interactions with columns:
                user, item, relevant, clicked_at, timestamp, rating, constant.
                clicked_at and timestamp are NaN for non-relevant items.
            rec_df (pd.DataFrame): Full recommendation slate with columns:
                user, item, rating (score). One row per (user, item) pair.
        """

        feedback_matrix = self.get_feedback_for_predictions(rec)
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
                USER_COL,
                ITEM_COL,
                "relevant",
                "clicked_at",
                TIMESTAMP_COL,
                RATING_COL,
                "constant",
            ],
        )

        interaction_df = interaction_df[interaction_df["relevant"] == 1.0]
        interaction_df["constant"] = 1.0  # For calibration purposes

        n_users = rec.shape[0]
        k = rec.shape[1]
        rec_df = pd.DataFrame(
            {
                USER_COL: torch.arange(n_users).repeat_interleave(k).numpy(),
                ITEM_COL: rec.reshape(-1).cpu().numpy(),
                "rating": score.reshape(-1).cpu().numpy(),
            }
        )

        return interaction_df, rec_df

    def bootstrap_clicks(self, k=20, num_interactions_bootstrapped=500_000):
        """
        Given unique users and unique items, recommend up to k items to every user
        using a preference matrix as a relevancy model and using a click model
        to simulate probability of user examinating an item.

        In order to ensure enough feedback data to train a model, we run the boostrap process
        until we have at least num_interactions_bootstrapped interactions, which is a
        hyperparameter that can be set when initializing the Simulator.

        """

        n_users = self.users.max() + 1

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
        with tqdm(
            total=num_interactions_bootstrapped, desc="Bootstrapping clicks"
        ) as pbar:
            while len(bootstrapped_df) < num_interactions_bootstrapped:
                mask = self._mask_previously_seen_items(bootstrapped_df).to(self.device)
                rec, score = random_rec(self.items, n_users, k, mask)
                round_df, _ = self.simulate_user_feedback(rec=rec, score=score)
                round_positives = len(round_df)
                bootstrapped_df = pd.concat(
                    [bootstrapped_df, round_df], ignore_index=True
                )
                pbar.update(round_positives)
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
                calibration_type=self.calibration_type,
            )
        return rec, score

    def _mask_previously_seen_items(self, users_history):
        mask = torch.ones((self.n_users, self.n_items), dtype=torch.float32)
        seen = users_history[[USER_COL, ITEM_COL]]
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
        weight_col = CALIBRATION_MODE_TO_COL_NAME.get(self.calibration_type, "constant")
        H_0 = boostrapped_df.copy()
        maces = []
        kl_divs = []
        initial_model = copy.deepcopy(self.model)

        user_history_tensor = build_user_genre_history_distribution(
            H_0,
            self.p_g_i,
            n_users=self.n_users,
            n_items=self.n_items,
            weight_col=weight_col,
        )

        for round_idx in tqdm(range(1, rounds + 1), desc="Processing rounds..."):
            # Sparsity = (Total Elements - Non-Zero Elements) / Total Elements
            sparsity = 1.0 - (
                torch.count_nonzero(self.oracle_tensor).item()
                / self.oracle_tensor.numel()
            )

            print(f"Sparsity at round {round_idx}: {sparsity:.2%}")
            mask = self._mask_previously_seen_items(boostrapped_df)
            rec, score = self._recommend(users_history=H_0, k=k, mask=mask)

            round_df, rec_df = self.simulate_user_feedback(
                rec=rec,
                score=score,
            )

            rec_genre_distribution_tensor = build_user_genre_history_distribution(
                df=rec_df,
                p_g_i=self.p_g_i,
                n_users=self.n_users,
                n_items=self.n_items,
                weight_col="rating",
            )
            iteration_mace = mace(
                rec_df=rec_df,
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
