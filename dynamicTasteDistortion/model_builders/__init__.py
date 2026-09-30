from functools import partial

from .bpr import build_bpr
from .knn import build_iknn, build_uknn
from .unpersonalized import build_most_popular

from recmodels.bpr_mf import bprMf, bprMFWithClickDebiasing

MODEL_BUILDERS = {
    "most_popular": build_most_popular,
    "bpr": partial(build_bpr, model_class=bprMFWithClickDebiasing),
    "bpr_classic": partial(build_bpr, model_class=bprMf),
    "iknn": build_iknn,
    "uknn": build_uknn,
}

bpr_param_grid = {
    "factors": [16, 32, 64, 128],
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "reg_lambda": [1e-5, 1e-4, 1e-3, 1e-2],
    "num_negatives": [1, 5, 10],
    "n_epochs": [5, 7, 10],
}


PARAM_GRIDS = {
    "bpr": bpr_param_grid,
    "bpr_classic": bpr_param_grid,
    "uknn": {"k_neighbors": [5, 10, 20, 50, 100]},
    "iknn": {"k_neighbors": [5, 10, 20, 50, 100]},
    "most_popular": {},
}
