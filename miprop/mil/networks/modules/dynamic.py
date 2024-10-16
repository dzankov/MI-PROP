import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import Softmax
from miqsar.estimators.neural_nets.base_nets import BaseRegressor, BaseClassifier, BaseNet
from miprop.mil.networks.modules.base import MainNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, alpha=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.alpha = alpha

    def forward(self, lengths, labels):
        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2
        margin_loss = labels * left + self.alpha * (1. - labels) * right
        return margin_loss.mean()


class Norm(nn.Module):
    def forward(self, inputs):
        lengths = torch.norm(inputs, p=2, dim=1, keepdim=True)
        return lengths


class Squash(nn.Module):
    def forward(self, inputs):
        norm = torch.norm(inputs, p=2, dim=2, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
        return scale * inputs


class DynamicPooling(nn.Module):
    def __init__(self, n_iter=3):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, x, m):
        x = m * x
        b = torch.zeros(x.shape[0], x.shape[1], 1).to(x.device)
        for t in range(self.n_iter):
            w = Softmax(dim=1)(m * b)
            w = torch.transpose(w, 2, 1)
            sigma = torch.bmm(w, x)
            s = Squash()(sigma)

            b_new = torch.sum(s * x, dim=2)
            b_new = b_new.reshape(b_new.shape[0], b_new.shape[1], 1)
            b = b + b_new
        w = Softmax(dim=1)(b)

        s = s.view(s.shape[0], s.shape[-1])
        w = w.view(w.shape[0], w.shape[1])
        return w, s


class DynamicPoolingNet(BaseNet):
    def __init__(self, ndim=None, init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.dynamic_pooling = DynamicPooling()
        self.estimator = Norm()

        if init_cuda:
            self.main_net.cuda()
            self.dynamic_pooling.cuda()
            self.estimator.cuda()
            
    def reset_weights(self):
        self.main_net.apply(self.reset_params)
        self.dynamic_pooling.apply(self.reset_params)
        self.estimator.apply(self.reset_params)

    def forward(self, x, m):
        x = self.main_net(x)
        w, s = self.dynamic_pooling(x, m)
        out = self.estimator(s)
        return w, out

    def predict(self, x):
        y_pred = super().predict(x)
        if isinstance(self, BaseRegressor):
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred


class DynamicPoolingNetRegressor(DynamicPoolingNet, BaseRegressor):
    def __init__(self, ndim=None, init_cuda=True):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

    def train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x), np.asarray(y)
        x, m = self.add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        if isinstance(self, BaseRegressor):
            self.scaler = MinMaxScaler()
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = self.scaler.transform(y_val.reshape(-1, 1)).flatten()

        x_train, y_train, m_train = self.array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self.array_to_tensor(x_val, y_val, m_val)
        return x_train, x_val, y_train, y_val, m_train, m_val


class DynamicPoolingNetClassifier(DynamicPoolingNet, BaseClassifier):
    def __init__(self, ndim=None, init_cuda=True):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

    def loss(self, y_pred, y_true):
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true.reshape(-1, 1))
        return loss

