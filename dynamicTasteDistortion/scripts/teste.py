import torch
from torch import Tensor
from tqdm import tqdm


def fragmentation(rec_tensor, s=0.9):
    """
    Fragmentation via RBO on item id overlap.
    Follows Vrijenhoek et al. (2021), based on Webber et al. (2010).

    rec_tensor : (n_users, k) ranked item indices
    """
    n_users, k = rec_tensor.shape

    rbo_scores = torch.zeros(n_users, n_users, device=rec_tensor.device)
    weight_sum = 0.0

    for d in range(1, k + 1):
        top_d = rec_tensor[:, :d]  # (n_users, d)
        matches = top_d.unsqueeze(1).unsqueeze(-1) == top_d.unsqueeze(0).unsqueeze(
            -2
        )  # (n_users, n_users, d, d)
        affinity = matches.any(dim=-1).float().sum(dim=-1) / d  # (n_users, n_users)
        rbo_scores += (s ** (d - 1)) * affinity
        weight_sum += s ** (d - 1)

    rbo_scores /= weight_sum
    mask = torch.triu(torch.ones(n_users, n_users, dtype=torch.bool), diagonal=1)
    return float(1.0 - rbo_scores[mask].mean())


if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    rec = torch.stack([torch.randperm(10000)[:20] for _ in range(1000)])
    t0 = time.time()
    print("fragmentation:", fragmentation(rec))

    print(f"time: {time.time() - t0:.1f}s")
    rec_popular = rec[0].unsqueeze(0).expand(1000, -1)
    print("fragmentation (rec popular):", fragmentation(rec_popular))
