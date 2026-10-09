import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from sklearn.metrics import f1_score

from surprise import SVD, NMF, SVDpp, Reader, Dataset as SurpriseDataset

from recmodels.utils.data import temporal_train_val_test_split
from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
)
MODEL_NAME_TO_CLASS_NAME = {"NMF": NMF, "SVD++": SVDpp}


class OracleChooser:
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




def choose_best_oracle(df, class_cutoff=4.0, rating_scale=(1, 5)):
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
        model_config = OracleChooser(model_name)
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
