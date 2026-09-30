from recmodels.evaluation import compute_map_at_k
import random
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from surprise import SVD, NMF, Reader, SVDpp, Dataset as SurpriseDataset
from itertools import product

from tqdm import tqdm, trange
import itertools
from recmodels.model import BaseModel
from recmodels.utils.data import temporal_train_val_test_split


from dynamicTasteDistortion.simulationConstants import (
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
    USER_COL,
    ITEM_COL,
)
from dynamicTasteDistortion.model_builders import MODEL_BUILDERS, PARAM_GRIDS

MODEL_NAME_TO_CLASS_NAME = {"NMF": NMF, "SVD++": SVDpp}


class MostPopularRecommender(BaseModel):

    def __init__(self, df):
        super().__init__()
        n_items = df.item.nunique()
        self.n_items = df.item.max() + 1

        pop_df = df.groupby(ITEM_COL).agg(
            popularity=(USER_COL, lambda group: len(group) / n_items)
        )

        max_item = pop_df.index.max()
        pop_tensor = torch.zeros(max_item + 1, dtype=torch.float32)

        pop_tensor[torch.tensor(pop_df.index.values)] = torch.tensor(
            pop_df["popularity"].values, dtype=torch.float32
        )
        self.item_2_popularity = pop_tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_2_popularity = self.item_2_popularity.to(self.device)

    def fit(self, train_df, debug):
        # No training needed for this model, as it relies solely on item popularity.
        pass

    def forward(self, users, items):
        # score depends only on item popularity
        return self.item_2_popularity[items]

    def evaluate(self, train_df, test_df, k=20):
        self.eval()

        train_pos = train_df.groupby("user")["item"].apply(set)
        test_pos = test_df.groupby("user")["item"].apply(set)
        # Make sure to use users present in both, specially important in CVTT
        # scenario
        eval_users = sorted(set(test_pos.index) & set(train_pos.index))

        all_items = torch.arange(self.n_items, device=self.device)

        with torch.no_grad():
            # Because most popular is not a personalized recommendation, we don´t need difference predictions
            # per user
            item_scores = self.forward(None, all_items)
            # we expand it to 2d in order to to top_k rank.
            score_matrix = item_scores.unsqueeze(0).expand(len(eval_users), -1).clone()

            # remove predictions for items exclusively in train dataset.
            for i, user_id in enumerate(eval_users):
                train_items = torch.tensor(list(train_pos[user_id]), device=self.device)
                score_matrix[i, train_items] = -torch.inf

            top_k = torch.topk(score_matrix, k=k, dim=1).indices.cpu().numpy()

        map_k = compute_map_at_k(top_k, eval_users, test_pos, k)

        self.train()
        return map_k


class HyperParameterTuner:
    def __init__(self, df, model, config=None, param_grid=None, seed=42):
        """
        model: key in MODEL_BUILDERS (e.g. "iknn", "bpr")
        config: optional config dict; its model_params act as fixed defaults
                that the tuned params override
        param_grid: optional override of PARAM_GRIDS[model]
        """
        if model not in MODEL_BUILDERS:
            raise ValueError(
                f"Unknown model '{model}'. Supported: {sorted(MODEL_BUILDERS)}"
            )
        self.df = df
        self.model_type = model
        self.config = config or {}
        self.param_grid = (
            PARAM_GRIDS.get(model, {}) if param_grid is None else param_grid
        )
        self.rng = random.Random(seed)
        builder = MODEL_BUILDERS[model]
        self.build_fn = lambda params: builder(self.config, self.df, **params)

    def _candidates(self, n_samples):
        names = list(self.param_grid)
        combos = [
            dict(zip(names, vals))
            for vals in itertools.product(*self.param_grid.values())
        ]
        return (
            combos if len(combos) <= n_samples else self.rng.sample(combos, n_samples)
        )

    def tune(self, truth_set, val_pct=0.2, test_pct=0.2, n_samples=20, k=20):
        if not self.param_grid:
            return self.build_fn({}), pd.DataFrame(), {}

        train_df, val_df, _ = temporal_train_val_test_split(
            df=self.df, user_col=USER_COL, val_pct=val_pct, test_pct=test_pct
        )
        pos_truth_set = truth_set[truth_set["relevant"] == 1].copy()

        candidates = self._candidates(n_samples)
        results = []
        for i, params in enumerate(tqdm(candidates, desc="Tuning")):
            model = self.build_fn(params)
            model.fit(train_df)
            score = model.evaluate(
                train_df=train_df, oot_df=val_df, oracle_df_pos=pos_truth_set, k=k
            )
            print(f"[{i+1}/{len(candidates)}] {params} -> MAP@{k}: {score:.4f}")
            results.append({"params": params, "map": score})

        best = max(results, key=lambda r: r["map"])
        results_df = pd.DataFrame(
            [{**r["params"], "map": r["map"]} for r in results]
        ).sort_values("map", ascending=False)

        return self.build_fn(best["params"]), results_df, best["params"]


# Oracle model based
class ModelChooser:
    def __init__(self, name):
        self.name = name
        self.models = {
            "SVD++": SVDpp,
            "SVD": SVD,
            "NMF": NMF,
        }

        self.param_grid_svd = {
            "n_factors": [15, 30, 100],
            "n_epochs": [10, 20, 30],
            "lr_all": [0.01, 0.002, 0.005],
            "reg_all": [0.02, 0.05, 0.1],
        }

        self.param_grid_nmf = {
            "n_factors": [15, 30, 100],
            "n_epochs": [50, 100],
            "reg_pu": [0.02, 0.05, 0.1],
            "reg_qi": [0.02, 0.05, 0.1],
        }

        self.model_name_to_params = {
            "SVD++": self.param_grid_svd,
            "NMF": self.param_grid_nmf,
            "SVD": self.param_grid_svd,
        }

        self.model = self.models[self.name]
        self.params = self.model_name_to_params[self.name]

    def yield_models(self, n_samples, seed=42):
        model = self.model
        params = self.params
        rng = np.random.default_rng(seed)

        samples = []
        for _ in trange(n_samples, desc="Sampling hyperparameters"):
            sampled_params = {
                key: rng.choice(values).item() for key, values in params.items()
            }
            samples.append((model(**sampled_params), sampled_params))

        return samples


def choose_best_model(df, class_cutoff=4.0, rating_scale=(1, 5)):
    f1_results = {}

    model_names = ["SVD", "SVD++", "NMF"]
    reader = Reader(rating_scale=rating_scale)

    train_df, val_df, test_df = temporal_train_val_test_split(
        df=df,
        user_col=USER_COL,
        val_pct=0.15,
        test_pct=0.15,
    )

    print(
        f"Separação de dataset baseado em tempo. Tamanho de treino, val e test: {len(train_df)}; {len(test_df)}; {len(val_df)}"
    )

    train_surprise = SurpriseDataset.load_from_df(
        train_df[["user", "item", "rating"]], reader
    )
    trainset_surprise = train_surprise.build_full_trainset()

    validation_set_surprise = list(
        val_df[["user", "item", "rating"]].itertuples(index=False, name=None)
    )
    test_set_surprise = list(
        test_df[["user", "item", "rating"]].itertuples(index=False, name=None)
    )

    # Maps the best hyperparamer variant
    family_winners = {}  # model_name -> (best_model, best_params, best_val_f1)

    for model_name in model_names:
        model_config = ModelChooser(model_name)
        models = model_config.yield_models(n_samples=40)

        best_score = float("-inf")
        best_model = None
        best_params = None

        for model, params in tqdm(models, desc=f"Optimizing {model_name}..."):
            model.fit(trainset_surprise)
            predictions = model.test(validation_set_surprise)

            y_pred = [1 if pred.est >= class_cutoff else 0 for pred in predictions]
            y_true = [1 if pred.r_ui >= class_cutoff else 0 for pred in predictions]
            val_f1 = f1_score(y_true, y_pred)

            if val_f1 > best_score:
                best_score = val_f1
                best_model = model
                best_params = str(params)

            f1_results[(model_name, str(params))] = {"val_f1": val_f1, "test_f1": None}

        family_winners[model_name] = (best_model, best_params, best_score)

    # Now we perform the final model selection on the test set
    global_best_score = float("-inf")
    global_best_model = None

    for model_name, (best_model, best_params, _) in family_winners.items():
        test_predictions = best_model.test(test_set_surprise)
        test_y_pred = [1 if pred.est >= 4 else 0 for pred in test_predictions]
        test_y_true = [1 if pred.r_ui >= 4 else 0 for pred in test_predictions]
        test_f1 = f1_score(test_y_true, test_y_pred)

        f1_results[(model_name, best_params)]["test_f1"] = test_f1

        if test_f1 > global_best_score:
            global_best_score = test_f1
            global_best_model = best_model

    rows = []
    for (model_name, params), scores in f1_results.items():
        rows.append(
            {
                "model_name": model_name,
                "params": params,
                "validation_f1": scores["val_f1"],
                "test_f1": scores["test_f1"],
                "is_winning_variant": scores["test_f1"] is not None,
            }
        )

    f1_df = pd.DataFrame(rows)
    return global_best_model, f1_df
