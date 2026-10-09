from recmodels.evaluation import compute_map_at_k
import torch


from sklearn.metrics import f1_score
from surprise import SVD, NMF, SVDpp, Dataset as SurpriseDataset

from recmodels.model import BaseModel


from dynamicTasteDistortion.simulationConstants import (
    USER_COL,
    ITEM_COL,
)


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
