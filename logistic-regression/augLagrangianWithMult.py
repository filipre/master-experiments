import torch
import torch.nn.functional as F

"""
\sum_k g_k(x_k) + \sum_k <y_k, x0 - xk> + \sum_k rho_k/2 ||x0 - xk||^2
checked
"""

def get(progress_loader, device, x0_model, x_models, y_models, rhos):
    augmented_lagrangian = 0
    number_nodes = len(x_models)

    with torch.no_grad():

        for k in range(number_nodes):

            # g_k(x) over "progress"
            # this is ok because the data has the same distribution over all g_k
            progress_loss = 0
            correct = 0
            for data, target in progress_loader:
                data, target = data.to(device), target.to(device)
                output = x_models[k](data)
                progress_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            progress_loss /= len(progress_loader.dataset)
            progress_acc = correct / len(progress_loader.dataset)
            print(f"[Node {k}] Loss: {progress_loss} ({(progress_acc * 100):.1f}%)")

            # <y_k, x_k - x>
            inner_product_x_x0 = {}
            for x0_name, x0_value in x0_model.named_parameters():
                for x_name, x_value in x_models[k].named_parameters():
                    for y_name, y_value in y_models[k].named_parameters():
                        if x0_name == x_name and x0_name == y_name:
                            # x_difference_vec = (x0_value - x_value).view(-1)
                            x_difference_vec = (x_value - x0_value).view(-1)
                            y_vec = y_value.view(-1)
                            inner_product_x_x0[x0_name] = y_vec.dot(x_difference_vec)

            # rho_k/2 | x_k - x |^2
            augmentation = {}
            for x0_name, x0_value in x0_model.named_parameters():
                for x_name, x_value in x_models[k].named_parameters():
                    if x0_name == x_name:
                        # norm2 = torch.norm(x0_value - x_value)**2
                        norm2 = torch.norm(x_value - x0_value)**2
                        augmentation[x0_name] = rhos[k]/2 * norm2

            # sum everything
            augmented_lagrangian = augmented_lagrangian + progress_loss
            for x0_name, x0_value in x0_model.named_parameters():
                augmented_lagrangian = augmented_lagrangian + augmentation[x0_name]

        progress_loss = 0
        correct = 0
        for data, target in progress_loader:
            data, target = data.to(device), target.to(device)
            output = x0_model(data)
            progress_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        progress_loss /= len(progress_loader.dataset)
        progress_acc = correct / len(progress_loader.dataset)
        print(f"[Node Master] Loss: {progress_loss} ({(progress_acc * 100):.1f}%)")

        print(f"Augmented Lagrangian: {augmented_lagrangian}")

    return augmented_lagrangian, progress_loss, progress_acc
