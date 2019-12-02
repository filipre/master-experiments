import torch

def solve(y_model, x0_model, x_model, rho):
    with torch.no_grad():

        for x0_name, x0_value in x0_model.named_parameters():
            for x_name, x_value in x_model.named_parameters():
                for y_name, y_value in y_model.named_parameters():
                    if x0_name == x_name and x0_name == y_name:
                        y_change = x0_value.data - x_value.data
                        y_value.data += rho * y_change # gradient ascent because we do maximization over yk

    return y_model
