import numpy as np
import torch

"""
w* = \argmin_{w\in R^L} \delta_\Delta(w) + \frac12 (w-v)^T H (w-v)
where H is diagonal (represented as array)
"""
def scaled_proj(v, H, device):
    N, L = v.size()
    Hv = H * v
    Hv_, ind = Hv.sort(dim=1, descending=True)
    ind_ = ind + torch.arange(start=0, end=N*L, step=L, device=device).view(N, 1).repeat(1, L)
    v_ = torch.take(v, ind_).view(N, L)
    H_ = torch.take(H.repeat(N, 1), ind_).view(N, L)
    sum_inv_H_ = torch.cumsum(torch.reciprocal(H_), -1)
    sum_v_ = torch.cumsum(v_, -1)
    potential_lambda = (torch.reciprocal(sum_inv_H_)) * (1-sum_v_)
    test = Hv_ + potential_lambda > 0
    range = torch.arange(1, L+1, device=device).repeat(N, 1)
    over_0 = test * range
    rho = torch.argmax(over_0, 1)
    rho_ = rho + torch.arange(start=0, end=N*L, step=L, device=device)
    right_lambda = torch.take(potential_lambda, rho_)
    right_lambda_ = right_lambda.view(N, 1).repeat(1, L)
    w = torch.max(torch.zeros_like(v, device=device), v + (right_lambda_ / H.repeat(N, 1)))
    return w

if __name__ == '__main__':
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(device)

    N, L = 5, 4
    v = torch.randn(N, L).to(device)
    H = torch.arange(start=1, end=L+1, dtype=torch.float).to(device)
    w = scaled_proj(v, H, device)
    print(w)
