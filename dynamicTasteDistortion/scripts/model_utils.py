import ast
import pickle
import os
from bprMf.evaluation import compute_map_at_k
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from surprise import SVD, NMF, Reader, SVDpp, Dataset as SurpriseDataset
from itertools import product

from tqdm import tqdm, trange


from bprMf.model import BaseModel
from bprMf.utils.data import temporal_train_val_test_split


from dynamicTasteDistortion.simulationConstants import (
    RESULTS_PATH,
    MODEL_ARTIFACTS_PATH,
    USER_COL,
    ITEM_COL,
)

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


bpr_param_grid = {
    "factors": [16, 32, 64, 128],
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "reg_lambda": [1e-5, 1e-4, 1e-3, 1e-2],
    "num_negatives": [1, 5, 10],
    "n_epochs": [5, 7, 10],
}


class HyperParameterTuner:
    def __init__(self, df, model, params=bpr_param_grid):
        self.params = params
        self.ModelClass = model
        self.seed = 42
        self.df = df
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_users = df.user.max() + 1
        self.n_items = df.item.max() + 1

    def tune(self, truth_set, val_pct=0.2, test_pct=0.2, n_samples=20, k=20):
        train_df, val_df, test_df = temporal_train_val_test_split(
            df=self.df,
            user_col=USER_COL,
            val_pct=val_pct,
            test_pct=test_pct,
        )
        pos_truth_set = truth_set[truth_set["rating"] == 1]
        rng = np.random.default_rng(self.seed)
        results = []
        for i in trange(n_samples, desc="Processing tuning rounds"):
            params = {
                "factors": int(rng.choice(self.params["factors"])),
                "lr": float(rng.choice(self.params["lr"])),
                "reg_lambda": float(rng.choice(self.params["reg_lambda"])),
                "num_negatives": int(rng.choice(self.params["num_negatives"])),
                "n_epochs": int(rng.choice(self.params["n_epochs"])),
            }

            print(f"[{i+1}/{n_samples}] Testing: {params}")
            model = self.ModelClass(
                num_users=self.n_users,
                num_items=self.n_items,
                dev=self.dev,
                **params,
            )
            model.fit(train_df)
            map_score = model.evaluate(
                train_df=train_df, oot_df=val_df, oracle_df_pos=pos_truth_set, k=k
            )

            print(f"  MAP@{k}: {map_score:.4f}")
            results.append({**params, "map": map_score})

        results_df = pd.DataFrame(results).sort_values("map", ascending=False)

        # retrain best model on train+val, evaluate on test
        best_params = results_df.iloc[0].drop("map").to_dict()
        best_params = {
            k: (int(v) if k != "lr" and k != "reg_lambda" else float(v))
            for k, v in best_params.items()
        }
        print(f"\nBest params: {best_params}")
        print(f"Best val MAP@{k}: {results_df.iloc[0]['map']:.4f}")

        train_val_df = pd.concat([train_df, val_df])
        final_model = self.ModelClass(
            num_users=self.n_users, num_items=self.n_items, dev=self.dev, **best_params
        )
        print(f"Training on train+val set...")
        final_model.fit(train_val_df)
        test_map = final_model.evaluate(
            train_df=train_val_df, oot_df=test_df, oracle_df_pos=pos_truth_set, k=k
        )
        print(f"Final test MAP@{k}: {test_map:.4f}")

        return final_model, results_df, best_params


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


def choose_best_model(df, class_cutoff=4.0):
    f1_results = {}

    model_names = ["SVD", "SVD++", "NMF"]
    reader = Reader(rating_scale=(1, 5))

    train_df, val_df, test_df = temporal_train_val_test_split(
        df=df,
        user_col=USER_COL,
        val_pct=0.15,
        test_pct=0.15,
    )

    print(f"Separação de dataset baseado em tempo. Tamanho de treino, val e test: {len(train_df)}; {len(test_df)}; {len(val_df)}")

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
