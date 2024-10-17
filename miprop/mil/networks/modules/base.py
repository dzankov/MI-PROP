import torch
import numpy as np
from torch import nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.nn import Sigmoid, Linear, ReLU, Sequential
from sklearn.model_selection import train_test_split
from miprop.mil.networks.modules.utils import MBSplitter


class EntropyRegularizer(nn.Module):
    def forward(self, w):
        ent = -1.0 * (w * w.log2()).sum(axis=1)
        reg = ent.mean()
        return reg


class BaseClassifier:
    def loss(self, y_pred, y_true):
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class MainNet:
    def __new__(cls, hidden_layer_sizes):
        inp_dim = hidden_layer_sizes[0]
        net = []
        for dim in hidden_layer_sizes[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net = Sequential(*net)
        return net


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


class BaseNet(nn.Module):
    def __init__(self,
                 hidden_layer_sizes=(128,),
                 num_epoch=500,
                 batch_size=128,
                 learning_rate=0.001,
                 weight_decay=0,
                 weight_dropout=0,
                 verbose=False,
                 init_cuda=True):

        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_dropout = weight_dropout
        self.batch_size = batch_size
        self.init_cuda = init_cuda
        self.verbose = verbose

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):
        pass

    def _reset_params(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x), np.asarray(y)
        x, m = add_padding(x)

        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        x_train, y_train, m_train = self._array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self._array_to_tensor(x_val, y_val, m_val)

        return x_train, x_val, y_train, y_val, m_train, m_val

    def _array_to_tensor(self, x, y, m):

        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.init_cuda:
            x, y, m = x.cuda(), y.cuda(), m.cuda()
        return x, y, m

    def _loss_batch(self, x_mb, y_mb, m_mb, optimizer=None):
        w_out, y_out = self.forward(x_mb, m_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, x, m):
        x = m * self.main_net(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(x)
        out = out.view(-1, 1)
        return None, out

    def fit(self, x, y):
        input_layer_size = x[0].shape[-1]
        self._initialize(input_layer_size=input_layer_size,
                         hidden_layer_sizes=self.hidden_layer_sizes,
                         init_cuda=self.init_cuda)

        x_train, x_val, y_train, y_val, m_train, m_val = self._train_val_split(x, y)
        optimizer = optim.Yogi(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        val_loss = []
        for epoch in range(self.num_epoch):
            mb = get_mini_batches(x_train, y_train, m_train, batch_size=self.batch_size)
            self.train()
            for x_mb, y_mb, m_mb in mb:
                loss = self._loss_batch(x_mb, y_mb, m_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self._loss_batch(x_val, y_val, m_val, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if self.verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)
        return self

    def predict(self, x):
        x, m = add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        return np.asarray(y_pred.cpu())

    def get_instance_weights(self, x):
        x, m = add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        w = w.view(w.shape[0], w.shape[-1]).cpu()
        w = [np.asarray(i[j.bool().flatten()]) for i, j in zip(w, m)]
        return w
