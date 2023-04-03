import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(X, y, batch_size, shuffle=False, drop_last=True):
    X_t = torch.Tensor(X)
    y_t = torch.Tensor(y)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader
