import torch
import torch.nn.functional as F

def solve(new_model, data, target, device, delayed_model, lr):
    data, target = data.to(device), target.to(device)
    output = delayed_model(data) # use delays for gradients
    loss = F.nll_loss(output, target, reduction='mean') # + reg_W + reg_b

    # SGD step
    loss.backward()

    with torch.no_grad():
        for x_name, x_value in new_model.named_parameters():
            for x_delayed_name, x_delayed_value in delayed_model.named_parameters():
                if x_name == x_delayed_name:
                    x_change = x_delayed_value.grad.data
                    x_value.data = x_value.data - lr * x_change

    delayed_model.zero_grad()

    return new_model, loss
