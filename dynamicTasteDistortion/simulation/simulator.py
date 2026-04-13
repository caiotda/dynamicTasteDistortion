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
    encode_interaction_matrix,
    build_interaction_timestamp_matrix,
    get_forget_probability,
    convert_item_genre_map_to_tensor,
    get_users_most_recent_interaction_timestamp,
    map_prediction_to_preferences,
    random_rec,
    update_genre_affinity_tensor,
    update_oracle_from_hits,
    calculate_preference_matrix,
)
from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
    GENRES_COL,
    TIMESTAMP_COL,
    RATING_COL,
)
from dynamicTasteDistortion.simulation.tensorUtils import (
    pandas_df_to_sparse_tensor,
)


from tqdm import tqdm

TS_NOW = pd.Timestamp.now().timestamp()


class Simulator:
    def __init__(
        self,
        oracle_matrix,
        model,
        user_timestamp_distribution,
        initial_date=TS_NOW,
        base_artifacts_path=None,
        num_interactions_bootstrapped=1_000_000,
        bootstrapped_df=None,
        calibration_type=None,
        preference_update_rate=0,
        compare_to_h_0=True,
    ):
        # TODO: documentação dos parametros
        self.device = (
            model.device
            if model is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Probability of acquiring new preferences from examined, but unclicked items
        self.preference_update_rate = preference_update_rate
        self.compare_to_h0 = compare_to_h_0
        # TODO: faz sentido esse parametro / valor do parametro?
        self.top_k_for_evaluation = 10
        self.calibration_type = calibration_type
        # Maps each user_id to their average timestamp between interactions probability
        # distribution. Timestamps deltas are sampled from it.
        self.timestamp_distribution = user_timestamp_distribution

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

        self.users = torch.tensor(users, device=self.device)

        self.items = torch.tensor(
            list(oracle_matrix[ITEM_COL].drop_duplicates()), device=self.device
        )

        # Check if we have a bootstrapped set of clicks set; if not, we run the bootstrapping
        # process.
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
        # Genre distribution per item.
        self.p_g_i = build_item_genre_distribution_tensor(
            self.click_matrix, self.n_items
        )
        n_genres = self.p_g_i.shape[1]

        # N_users x N_items matrix that tracks the most recent interaction between each
        # user and item in the simulation.
        self.interaction_recency_matrix = build_interaction_timestamp_matrix(
            self.n_users, self.n_items, self.oracle_tensor, initial_date, self.device
        )

        # Builds a multi hot encoding tensor of shape n_items x n_genres.
        self.genre_tensor = convert_item_genre_map_to_tensor(self.item2genreMap)

        # Stores the probability of each user forgetting each item. This is time dependant
        # And depends on the genre affinity between the user and the item.
        self.forgetting_probability = torch.ones_like(self.interaction_recency_matrix)
        self.genre_affinity = torch.zeros(self.n_users, n_genres, device=self.device)

        # Basic persistent configurations
        if base_artifacts_path is not None and not os.path.exists(base_artifacts_path):
            os.makedirs(base_artifacts_path)

    def update_user_model(self, predictions, feedback_matrix, users_ids, clicked_items):
        """
        Updates the simulated users' preference model based on interacted items in a
        recommendation tensor.

        Args:
            predictions: Recommended items to evaluate.
            feedback_matrix: User feedback on recommendations.
            users_ids: User identifiers being updated.
            clicked_items: Items that users clicked/interacted with.
        """
        hit_matrix = calculate_preference_matrix(
            oracle_tensor=self.oracle_tensor,
            interaction_matrix=feedback_matrix,
            recommendation_list=predictions,
            preference_update_rate=self.preference_update_rate,
            preference_forgetting_probability=self.forgetting_probability,
        )

        self.oracle_tensor = update_oracle_from_hits(
            self.oracle_tensor, predictions, hit_matrix
        )

        user_tensor = torch.tensor(users_ids, device=self.device)
        self.genre_affinity, G = update_genre_affinity_tensor(
            user_tensor, self.genre_affinity, clicked_items, self.genre_tensor
        )
        self.forgetting_probability = get_forget_probability(
            self.interaction_recency_matrix, G
        )

    def get_user_feedback_from_predictions(self, recommendation_tensor):
        """
        Returns which items in the recommendation tensor were interacted by the simulated users

        Args:
            recommendation_tensor (torch.tensor): items recommended to each users (n_users, k)

        Returns:
            interaction_matrix (torch.tensor): binary matrix of shape (n_users, k) where
            each entry encodes wether the n-th user clicked on the k-th item in the recommendation
        """
        assert (recommendation_tensor >= 0).all(), "Item IDs must be non-negative"
        hit_matrix = map_prediction_to_preferences(
            self.oracle_tensor, recommendation_tensor
        )

        interaction_matrix = encode_interaction_matrix(
            recommendation_tensor, hit_matrix
        )
        return interaction_matrix

    def simulate_user_feedback(self, rec, score, from_bootstrap=False):
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
        should_update_user_model = not from_bootstrap
        feedback_matrix = self.get_user_feedback_from_predictions(rec)
        # We retrieve only clicked interactions, flagged as 1
        indices = torch.nonzero(feedback_matrix == 1, as_tuple=False)
        users_indices, click_positions = indices[:, 0].tolist(), indices[:, 1].tolist()
        # We retrieved the clicked items by using the user_ids with interaction
        # alongside the click positions
        clicked_items = rec[users_indices, click_positions]
        user_ids = [self.user_idx_to_id[idx] for idx in users_indices]

        feedbacks = feedback_matrix.flatten().tolist()
        items = clicked_items.flatten().tolist()
        scores = score.flatten().tolist()
        constant = [1.0] * len(scores)
        timestamps = [
            (self.timestamp_distribution[user].rvs(1)[0] / 60)
            + get_users_most_recent_interaction_timestamp(
                self.interaction_recency_matrix, user
            )
            for user in user_ids
        ]
        timestamps_tensor = torch.tensor(
            timestamps, device=self.device, dtype=self.interaction_recency_matrix.dtype
        )

        self.interaction_recency_matrix[user_ids, items] = timestamps_tensor
        # Update interest retention given new timestamps

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
        # We're only interested in updating the user model on the simulation, not on the bootstrapping
        # step. Also, if we don´t pass the preference_update_rate parameter, we assume that
        # we are simulating the baseline case: user preference remains static through time.
        if should_update_user_model and self.preference_update_rate != 0:
            self.update_user_model(
                predictions=rec,
                feedback_matrix=feedback_matrix,
                users_ids=user_ids,
                clicked_items=clicked_items,
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
                round_df, _ = self.simulate_user_feedback(
                    rec=rec, score=score, from_bootstrap=True
                )
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

        bootstrapped_df = self.click_matrix.copy()
        H_0 = bootstrapped_df.copy()
        maces = []
        kl_divs = []
        # This ensures that we always have a fresh model at each retrain, without knowing
        # its parameters
        initial_model = copy.deepcopy(self.model)

        # Builds the tensor that tells the importance of its genre according to the users history
        # (H0) and the importance strategy used (weight col)
        user_history_tensor = build_user_genre_history_distribution(
            H_0,
            self.p_g_i,
            n_users=self.n_users,
            n_items=self.n_items,
            weight_col=(
                "constant" if self.calibration_type is None else self.calibration_type
            ),
        )
        mask = None
        bootstrapped_df = pd.DataFrame({}, columns=bootstrapped_df.columns)
        for round_idx in tqdm(range(1, rounds + 1), desc="Processing rounds..."):
            # We avoid recommending repeated items in the same interaction.
            rec, score = self._recommend(users_history=H_0, k=k, mask=mask)

            round_df, rec_df = self.simulate_user_feedback(
                rec=rec,
                score=score,
            )

            # We calculate the taste distortion between what's being recommended and the users initial taste
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

            # What was interacted with (round_df) gets added to the running click df.
            bootstrapped_df = pd.concat([bootstrapped_df, round_df], ignore_index=True)

            # At every L rounds, we retrain the model and reset the accumulated clicks, which is done
            # to avoid an ever growing set of clicks to train the model on.
            mask = self._mask_previously_seen_items(bootstrapped_df)

            if round_idx % L == 0:
                if not self.compare_to_h0:
                    user_history_tensor = build_user_genre_history_distribution(
                        bootstrapped_df,
                        self.p_g_i,
                        n_users=self.n_users,
                        n_items=self.n_items,
                        weight_col=(
                            "constant"
                            if self.calibration_type is None
                            else self.calibration_type
                        ),
                    )
                if not self.use_random_rec:
                    print("retraining model...")
                    self.model = copy.deepcopy(initial_model)
                    self.model.to(initial_model.device)

                    _ = self.model.fit(bootstrapped_df, debug=False)

        return bootstrapped_df, maces, kl_divs
