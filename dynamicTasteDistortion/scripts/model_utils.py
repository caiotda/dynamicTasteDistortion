import ast
import pickle
import os
from bprMf.bprMf.evaluation import average_precision_at_k, compute_map_at_k
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from surprise import NMF, Reader, SVDpp, Dataset as SurpriseDataset
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

    def tune(self, val_pct=0.1, test_pct=0.1, n_samples=20, k=20):
        train_df, val_df, test_df = temporal_train_val_test_split(
            df=self.df,
            user_col=USER_COL,
            val_pct=val_pct,
            test_pct=test_pct,
        )
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
            map_score = model.evaluate(train_df=train_df, test_df=val_df, k=k)

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
        test_map = final_model.evaluate(train_df=train_val_df, test_df=test_df, k=k)
        print(f"Final test MAP@{k}: {test_map:.4f}")

        return final_model, results_df, best_params


# Oracle model based
class ModelChooser:
    def __init__(self, name):
        self.name = name
        self.models = {
            "SVD++": SVDpp,
            "NMF": NMF,
        }

        self.param_grid_svd = {
            "n_epochs": [10, 20],
            "lr_all": [0.002, 0.005],
            "reg_all": [0.02, 0.1],
        }

        self.param_grid_nmf = {
            "n_factors": [15, 30],
            "n_epochs": [50, 100],
            "reg_pu": [0.06, 0.1],
            "reg_qi": [0.06, 0.1],
        }

        self.model_name_to_params = {
            "SVD++": self.param_grid_svd,
            "NMF": self.param_grid_nmf,
        }

        self.model = self.models[self.name]
        self.params = self.model_name_to_params[self.name]

    def yield_models(self):
        model = self.model
        params = self.params
        param_names = list(params.keys())
        combinations = list(product(*params.values()))
        dicts = [dict(zip(param_names, values)) for values in combinations]
        return [(model(**param), param) for param in dicts]


def choose_best_model(df, data_type):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_results = {}

    model_names = ["SVD++", "NMF"]
    reader = Reader(rating_scale=(1, 5))

    usuarios = df["user"].unique()

    for model_name in model_names:
        model_config = ModelChooser(model_name)
        models = model_config.yield_models()

        for model, params in tqdm(
            models, desc=f"Starting optimization for model {model_name}..."
        ):
            f1_scores = []

            for train_users_idx, test_users_idx in kf.split(usuarios):
                train_users = set(usuarios[train_users_idx])
                test_users = set(usuarios[test_users_idx])

                trainset = df[df.user.isin(train_users)]
                testset = df[df.user.isin(test_users)]

                train_surprise = SurpriseDataset.load_from_df(
                    trainset[["user", "item", "rating"]], reader
                )
                trainset_surprise = train_surprise.build_full_trainset()

                testset_surprise = list(
                    testset[["user", "item", "rating"]].itertuples(
                        index=False, name=None
                    )
                )

                model.fit(trainset_surprise)
                predictions = model.test(testset_surprise)

                y_pred = [1 if pred.est >= 4 else 0 for pred in predictions]
                y_true = [1 if pred.r_ui >= 4 else 0 for pred in predictions]

                f1_scores.append(f1_score(y_true, y_pred))

            f1_results[(model_config, str(params))] = np.mean(f1_scores)

    f1_df = pd.DataFrame(
        [
            {"model": str(model_config.name), "params": params, "f1_score": score}
            for (model_config, params), score in f1_results.items()
        ]
    )
    destination_dir = f"{RESULTS_PATH}/{data_type}"
    model_artifacts_path = f"{MODEL_ARTIFACTS_PATH}/{data_type}"
    # Mover isso pra main?
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not os.path.exists(model_artifacts_path):
        os.makedirs(model_artifacts_path)

    f1_df.to_pickle(f"{destination_dir}/oracle_model_f1_results.pkl")

    best_results = f1_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model = best_results.model
    best_params = ast.literal_eval(best_results.params)

    model_class = MODEL_NAME_TO_CLASS_NAME[best_model]
    oracle_model = model_class(**best_params)
    artifact = {"params": best_params, "model_class": model_class}
    with open(f"{MODEL_ARTIFACTS_PATH}/{data_type}/oracle_model_params.pkl", "wb") as f:
        pickle.dump(artifact, f)

    return oracle_model
