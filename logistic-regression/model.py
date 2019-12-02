import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.view(-1, 28*28)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

def copyWeights(fromModel, toModel):
    # TODO: get list of names, then iterate over that
    for paramName, paramValue, in fromModel.named_parameters():
        for netCopyName, netCopyValue, in toModel.named_parameters():
            if paramName == netCopyName:
                netCopyValue.data = paramValue.data.clone()

def copyGrads(fromModel, toModel):
    # TODO: get list of names, then iterate over that
    for paramName, paramValue, in fromModel.named_parameters():
        for netCopyName, netCopyValue, in toModel.named_parameters():
            if paramName == netCopyName:
                netCopyValue.grad = paramValue.grad.clone()
