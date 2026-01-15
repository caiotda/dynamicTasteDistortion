import torch
from dynamicTasteDistortion.simulationConstants import USER_COL, ITEM_COL


def get_matrix_coordinates(matrix):
    return torch.nonzero(torch.ones_like(matrix), as_tuple=False)


def pandas_df_to_sparse_tensor(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings_tensor = torch.tensor(
        df[[USER_COL, ITEM_COL, "rating"]].to_numpy(), device=device
    )
    n_users = ratings_tensor[:, 0].max().item() + 1
    n_items = ratings_tensor[:, 1].max().item() + 1

    ratings_mat = torch.zeros(
        (n_users, n_items), device=ratings_tensor.device, dtype=ratings_tensor.dtype
    )

    users = ratings_tensor[:, 0]
    items = ratings_tensor[:, 1]
    ratings = ratings_tensor[:, 2]

    ratings_mat[users, items] = ratings

    return ratings_mat
