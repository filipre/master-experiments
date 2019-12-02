"""
x_k = argmin_{x_k} g_k(x_k) + <y_k, x0 - x_k> + \rho/2 ||x0 - x_k||^2

\nabla g_k(x_k) - y_k - \rho (x0 - x_k)
"""

import torch.nn.functional as F
import torch
import numpy as np

def solve(xk_model, loader, device, x0_model, yk_model, rho, lr, epochs):
    print(f"Loader Size: {len(loader.dataset)}")

    # scores, losses, residuals = [], [], []

    for worker_epoch in range(epochs):
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            output = xk_model(data)
            loss = F.nll_loss(output, target, reduction='mean')
            loss.backward()

            with torch.no_grad():
                for x0_name, x0_value in x0_model.named_parameters():
                    for x_name, x_value in xk_model.named_parameters():
                        for y_name, y_value in yk_model.named_parameters():
                            if x0_name == x_name and x0_name == y_name:
                                x_change = x_value.grad.data - y_value.data + rho * (x_value.data - x0_value.data) # same: - rho * (x0_value.data - x_value.data)
                                x_value.data -= lr * x_change

                xk_model.zero_grad()

                # calculate metrics
                # loss (already calculated)
                # losses.append(loss.item())

                # L1 residuals between x_k and x0
                # residual = 0
                # for x0_name, x0_value in x0_model.named_parameters():
                #     for x_name, x_value in xk_model.named_parameters():
                #         if x0_name == x_name:
                #             residual = residual + np.linalg.norm(x_value.data - x0_value.data, ord=1)
                # residuals.append(residual)

                # local cost function g_k(x_k) + rho/2 ||x0 - x_k||
                # score = 0
                # score = score + loss
                # augmentation = {}
                # for x0_name, x0_value in x0_model.named_parameters():
                #     for x_name, x_value in xk_model.named_parameters():
                #         if x0_name == x_name:
                #             norm2 = torch.norm(x_value - x0_value) ** 2
                #             augmentation[x0_name] = rho/2 * norm2
                # for x0_name, x0_value in x0_model.named_parameters():
                #     score = score + augmentation[x0_name]
                # scores.append(score.item())

                # Accuracy
                # correct = 0
                # pred = output.argmax(dim=1, keepdim=True)
                # correct += pred.eq(target.view_as(pred)).sum().item()
                # acc = correct / len(data)

            # scores.append(score.item())
            # losses.append(loss.item())
            # print(f"[{worker_epoch}] Score: {score.item()}, Loss: {loss.item()}, Acc: {(acc * 100):.1f}%")

    # return xk_model, scores, losses, residuals
    return xk_model
