import itertools
import random

import numpy as np
import pandas as pd

from tqdm import tqdm


from recmodels.utils.data import temporal_train_val_test_split
from dynamicTasteDistortion.model_builders import MODEL_BUILDERS, PARAM_GRIDS

from dynamicTasteDistortion.simulationConstants import USER_COL


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
            print(f"Params: {params}")
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
