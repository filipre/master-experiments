import numpy as np
import torch
from scipy import sparse

def gradient_operator(ny, nx, full_ny=None, full_nx=None):
    if full_ny == None:
        full_ny = ny
    if full_nx == None:
        full_nx = nx
    n = ny*nx
    a = torch.arange(start=0, end=n-ny)
    b = torch.arange(start=ny, end=n)
    c = torch.arange(start=n, end=2*n)
    d = torch.arange(start=0, end=n)
    e = c[0:-1]
    f = torch.arange(start=1, end=n)
    ones = torch.ones(n-ny)
    b1_ones = ones
    b1_neg_ones = -ones
    g = torch.cat((torch.ones(ny-1), torch.zeros(1)), 0).repeat(nx)
    b2_ones = g[0:-1]
    b2_neg_ones = -1 * g
    y_coord = torch.cat((a, a, c, e), 0)
    x_coord = torch.cat((a, b, d, f), 0)
    ind = torch.stack((y_coord, x_coord))
    val = torch.cat((b1_neg_ones, b1_ones, b2_neg_ones, b2_ones), 0)
    operator = torch.sparse.FloatTensor(ind, val, torch.Size([2*n, n]))
    normalization_h = 1 / np.sqrt(ny * nx) # 1 / np.sqrt(full_ny * full_nx)
    return operator / np.sqrt(ny * nx) # normalization_h * operator

def selection_matrix(m, n, k=0):
    assert isinstance(m, int), "m is not an integer"
    assert isinstance(n, int), "n is not an integer"
    assert isinstance(k, int), "k is not an integer"
    assert m <= n, "m musn't be larger than n"
    assert k >= 0, "k must be positive"
    y_coord = torch.arange(start=0, end=m)
    x_coord = torch.arange(start=k, end=k+m)
    ind = torch.stack((y_coord, x_coord))
    val = torch.ones(m)
    matrix = torch.sparse.IntTensor(ind, val, torch.Size([m, n]))
    return matrix

if __name__ == '__main__':
    D = gradient_operator(100, 200)
    print(D)

    A1 = selection_matrix(3, 9, 1)
    print(A1.to_dense())
