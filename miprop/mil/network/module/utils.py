import numpy as np
import torch
from torch import nn
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import DataLoader, Dataset


class Extractor:
    def __new__(cls, hidden_layer_sizes):
        inp_dim = hidden_layer_sizes[0]
        net = []
        for dim in hidden_layer_sizes[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net = Sequential(*net)
        return net


class Pooling(nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()
        self.pool = pool

    def forward(self, x, m):
        x = m * x
        if self.pool == 'mean':
            out = x.sum(axis=1) / m.sum(axis=1)
        elif self.pool == 'max':
            out = x.max(dim=1)[0]
        elif self.pool == 'lse':
            out = x.exp().sum(dim=1).log()
        return out

    def extra_repr(self):
        return 'Pooling(out_dim=1)'


def add_padding(x):
    bag_size = max(len(i) for i in x)
    mask = np.ones((len(x), bag_size, 1))

    out = []
    for i, bag in enumerate(x):
        bag = np.asarray(bag)
        if len(bag) < bag_size:
            mask[i][len(bag):] = 0
            padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
            bag = np.vstack((bag, padding))
        out.append(bag)
    out_bags = np.asarray(out)
    return out_bags, mask


def get_mini_batches(x, y, m, batch_size=16):
    data = MBSplitter(x, y, m)
    mb = DataLoader(data, batch_size=batch_size, shuffle=True)
    return mb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MBSplitter(Dataset):
    def __init__(self, x, y, m):
        super(MBSplitter, self).__init__()
        self.x = x
        self.y = y
        self.m = m

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]

    def __len__(self):
        return len(self.y)
