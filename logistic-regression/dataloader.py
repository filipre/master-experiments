import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def getSplittedTrainingLoaders(number_nodes, batch_size, kwargs, partial=None):
    loaders = []
    train_data = _getTrainDataset(partial)
    equal_split = _makeEqualSplit(len(train_data), number_nodes)
    dataset_split = random_split(train_data, equal_split)
    for worker, dataset_subset in enumerate(dataset_split):
        loader = torch.utils.data.DataLoader(dataset_subset, batch_size=batch_size, shuffle=True, **kwargs)
        loaders.append(loader)
    return loaders

def getSameTrainingLoaders(number_nodes, batch_size, kwargs, partial=None):
    loaders = []
    train_data = _getTrainDataset(partial)
    for worker in range(number_nodes):
        loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
        loaders.append(loader)
    return loaders

def getProgressLoader(kwargs, batch_size=1000, progress_size=1000):
    test_data = _getTestDataset()
    test_size = len(test_data) - progress_size
    _, progress_dataset = random_split(test_data, [test_size, progress_size])
    progress_loader = torch.utils.data.DataLoader(progress_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return progress_loader

def getTestLoader(kwargs, batch_size=1000):
    test_data = _getTestDataset()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_data

def _getTrainDataset(partial=None):
    mnist_train_data = datasets.MNIST('', train=True, download=True, transform=mnist_transform)
    if partial is None:
        partial = len(mnist_train_data)
    partial_train_data, _ = random_split(mnist_train_data, [partial, len(mnist_train_data) - partial])
    return partial_train_data

def _getTestDataset():
    return datasets.MNIST('', train=False, download=True, transform=mnist_transform)

def _makeEqualSplit(n, k):
    split = [n//k] * k
    remainder = n - k*(n//k)
    for i in range(remainder):
        split[i] = split[i] + 1
    return split

# def _batchGenerator(dataloader, max_epoch, worker):
#     for epoch in range(1, max_epoch+1):
#         print(f"[Worker {worker}] Current epoch: {epoch}")
#         for data, target in dataloader:
#             yield data, target
