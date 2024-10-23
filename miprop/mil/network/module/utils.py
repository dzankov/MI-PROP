import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


