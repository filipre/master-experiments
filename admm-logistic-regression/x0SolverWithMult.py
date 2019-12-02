"""
\sum_k g_k(x_k) + \sum_k <y_k, x_0 - x_k> + \sum_k \tau_k/2 ||x_0 - x_k||^2

x_0 = ()\sum_k \tau_k x_k - \sum_k y_k) / \sum_k \tau_k
"""


import torch

def solve(x0_model, x_models, y_models, rhos):
    assert len(x_models) == len(y_models), "x and y models not same number nodes"
    number_nodes = len(x_models)

    with torch.no_grad():

        rho_x_sum = {}
        for k in range(number_nodes):
            for x_name, x_value in x_models[k].named_parameters():
                rho_x = rhos[k] * x_value
                rho_x_sum[x_name] = rho_x if x_name not in rho_x_sum else rho_x_sum[x_name] + rho_x

        y_sum = {}
        for k in range(number_nodes):
            for y_name, y_value in y_models[k].named_parameters():
                y_sum[y_name] = y_value if y_name not in y_sum else y_sum[y_name] + y_value

        rho_sum = sum(rhos)

        for x0_name, x0_value in x0_model.named_parameters():
            x0_value.data = (rho_x_sum[x0_name] - y_sum[x0_name]) / rho_sum

    return x0_model
