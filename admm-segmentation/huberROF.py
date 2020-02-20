# def huber(x, delta):
#     return np.where(np.abs(x) <= delta, (x**2)/(2*delta), np.abs(x) - delta/2)
#
# def diff_huber(x, delta):
#     return np.where(np.abs(x) <= delta, x/delta, np.sign(x))

"""
argmin_u \alpha ||Du||_{1,\delta} + \tau/2 || u - v ||^2
"""

import torch
import time

def huber(x, delta):
    return torch.where(torch.le(torch.abs(x), delta), torch.pow(x, 2)/(2*delta), torch.abs(x) - delta/2)

def objective(u, D, v, alpha, delta, tau):
    Du = torch.sparse.mm(D, u) # vector
    return alpha * torch.sum(huber(Du, delta)) + tau/2 * torch.pow(torch.norm(u - v), 2)

def sigma(x, delta):
    return torch.where(torch.le(torch.abs(x), delta), 1.0/delta, torch.sign(x)/x)
    # return torch.where(torch.le(x, delta), 1/delta, torch.sign(x)/x)

# def solver_L(x, A, G, tau, beta, delta):
#     b = sigma(G*x, delta)
#     # return 2*A.T*A + beta*G.T*sparse.diags(b)*G
#     return tau*A.T*A + beta*G.T*sparse.diags(b)*G
def solver_Au(u, D, alpha, delta, tau):
    Du = torch.sparse.mm(D, u)
    b = sigma(Du, delta)
    bDu = b * Du
    DTbDu = torch.sparse.mm(D.t(), bDu)
    return tau*u + alpha*DTbDu

def solver_b(v, tau):
    return tau*v

# tau/2 ||u - v||^2 + \alpha ||Du||_{1,\delta}
def opt_condition(u, D, v, alpha, delta, tau):
    opt_cond = torch.zeros_like(u)
    _, L = opt_cond.size()
    for l in range(L):
        u_l = u[:, l].view(-1, 1)
        v_l = v[:, l].view(-1, 1)
        solv_Au = solver_Au(u_l, D, alpha, delta, tau)
        solv_b = solver_b(v_l, tau)
        opt_cond[:, l] = (solv_Au - solv_b).view(-1)
    return opt_cond


def solve(D, v, L2, device, alpha=1, delta=1, tau=1, theta=1, max_iter=100000, tol=1e-10, verbose=False):
    # assume that they are already on device
    # D = D.to(device)
    # v = v.to(device)
    alpha = torch.tensor(alpha).to(device)
    delta = torch.tensor(delta).to(device)
    tau = torch.tensor(tau).to(device)

    m, n = D.size()
    n_, L = v.size()
    assert n == n_, "D and v dimensions do not match"
    if verbose is False:
        verbose = max_iter

    lr_primal = 1
    lr_dual = 1 / (L2 * lr_primal)
    # print("LR Primal and Dual:", lr_primal, lr_dual)

    p = torch.rand(m, L).to(device)
    u = torch.rand(n, L).to(device)
    ubar = torch.rand(n, L).to(device)
    obj = objective(u, D, v, alpha, delta, tau)
    opt = opt_condition(u, D, v, alpha, delta, tau)

    alphas = alpha * torch.ones_like(p)
    alphas = alphas.to(device)

    for t in range(max_iter):
        # dual update
        Du = torch.sparse.mm(D, ubar)
        p_snake = p + lr_dual * Du
        p_huber = p_snake / (1 + lr_dual * delta)
        p_new = p_huber / torch.max(alphas, torch.abs(p_huber))

        # primal update
        DTp = torch.sparse.mm(D.t(), p_new) # DTp = torch.sparse.mm(torch.t(D), p)
        u_snake = u - lr_primal * DTp
        u_new = (u_snake + lr_primal * tau * v) / (1 + lr_primal * tau)

        # primal bar update
        ubar_new = u_new + theta * (u_new - u)

        # Evaluation
        obj_new = objective(u_new, D, v, alpha, delta, tau)
        opt_new = opt_condition(u_new, D, v, alpha, delta, tau)

        # if t % verbose == 0:
        #     print(f"[{t}] {obj} {torch.norm(opt_new, p=1)}")

        if obj_new > obj:
            lr_primal = lr_primal / 2
            lr_dual = 1 / (L2 * lr_primal)
            # print("Warning: Objective is increasing instead of decreasing. Decreasing LR and trying again: ", lr_primal, lr_dual)
            continue

        u = u_new
        p = p_new
        ubar = ubar_new

        # tols = 1e-6 * torch.ones_like(opt_new)
        # if torch.all(torch.abs(opt - opt_new) < tols):
        #     print("Done because not progressing anymore")
        #     break

        if torch.abs(obj - obj_new) < tol:
            # print(f"[{t}] {obj} {torch.norm(opt_new, p=1)}")
            # print("Done because not progressing anymore")
            break

        obj = obj_new
        opt = opt_new

    return u


if __name__ == '__main__':
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(device)

    x = 20
    y = 5
    L = 1
    n = x*y
    u = torch.rand(n, L)
    D = torch.randn(2*n, n).to_sparse()
    v = torch.rand(n, L)

    L2 = 8 / n
    alpha = 0.1
    delta = 0.01
    tau = 1.0
    u_opt = solve(D, v, L2, device, alpha=alpha, delta=delta, tau=tau, theta=1, tol=1e-9, verbose=1, max_iter=10000)
    print("\n", u_opt)

    # u_opt2 = torch.zeros_like(u_opt)
    # for l in range(L):
    #     v_l = v[:, l].view(-1, 1)
    #     sol = solve(D, v_l, alpha=10, delta=0.1, tau=5, lr_primal=0.0002, theta=1, tol=1e-9, verbose=100, max_iter=10000)
    #     u_opt2[:, l] = sol.view(-1)
    # print("u_opt:", objective(u_opt, D, v, 10, 0.1, 5))
    # print("u_opt2:", objective(u_opt2, D, v, 10, 0.1, 5))
