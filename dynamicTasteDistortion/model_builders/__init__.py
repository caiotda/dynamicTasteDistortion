from functools import partial

from .bpr import build_bpr
from .knn import build_iknn, build_uknn
from .unpersonalized import build_most_popular

from recmodels.recmodels.bpr_mf import bprMf, bprMFWithClickDebiasing

MODEL_BUILDERS = {
    "most_popular": build_most_popular,
    "bpr": partial(build_bpr, model_class=bprMFWithClickDebiasing),
    "bpr_classic": partial(build_bpr, model_class=bprMf),
    "iknn": build_iknn,
    "uknn": build_uknn,
}
