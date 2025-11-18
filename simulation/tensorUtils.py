import torch



def get_matrix_coordinates(matrix):
    return torch.nonzero(torch.ones_like(matrix), as_tuple=False)