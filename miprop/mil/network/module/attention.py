import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh, ReLU
from torch.nn.functional import softmax
from miprop.mil.network.module.base import BaseClassifier, BaseNetwork, MainNetwork


class AttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNetwork((input_layer_size, *hidden_layer_sizes))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        #
        input_dim = hidden_layer_sizes[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(ReLU())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)

        w = InstanceWeightDropout(p=self.instance_weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class TemperatureAttentionNetwork(AttentionNetwork):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, m):
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / self.instance_weight_dropout)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GlobalTemperatureAttentionNetwork(AttentionNetwork):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, m):
        temp = torch.tensor(self.instance_weight_dropout).to(x.device)
        temp.to(x.device)

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / temp)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GumbelAttentionNetwork(AttentionNetwork):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, m):

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        if self.instance_weight_dropout == 0:
            w = Softmax(dim=2)(x_det)
        else:
            w = nn.functional.gumbel_softmax(x_det, tau=self.instance_weight_dropout, dim=2)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)

        return w, out


class GatedAttentionNetwork(AttentionNetwork, BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNetwork((input_layer_size, *hidden_layer_sizes))
        self.attention_V = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Tanh())
        self.attention_U = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Sigmoid())
        self.detector = Linear(det_ndim[0], 1)
        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if init_cuda:
            self.main_net.cuda()
            self.attention_V.cuda()
            self.attention_U.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        w_v = self.attention_V(x)
        w_u = self.attention_U(x)

        x_det = torch.transpose(m * self.detector(w_v * w_u), 2, 1)
        w = Softmax(dim=2)(x_det)
        # w = InstanceWeightDropout(p=self.instance_weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class SelfAttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNetwork((input_layer_size, *hidden_layer_sizes))
        self.self_attention = SelfAttention(hidden_layer_sizes[-1], hidden_layer_sizes[-1])
        self.detector = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Tanh(), Linear(det_ndim[0], 1))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if init_cuda:
            self.main_net.cuda()
            self.self_attention.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        x = self.self_attention(x)

        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)
        # w = InstanceWeightDropout(p=self.instance_weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class SelfAttention(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.w_query = nn.Linear(inp_dim, out_dim)
        self.w_key = nn.Linear(inp_dim, out_dim)
        self.w_value = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        keys = self.w_key(x)
        querys = self.w_query(x)
        values = self.w_value(x)

        att_weights = softmax(querys @ torch.transpose(keys, 2, 1), dim=-1)
        weighted_values = values[:, :, None] * torch.transpose(att_weights, 2, 1)[:, :, :, None]
        outputs = weighted_values.sum(dim=1)

        return outputs


class InstanceWeightDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, w):
        if self.p == 0:
            return w
        d0 = [[i] for i in range(len(w))]
        d1 = w.argsort(dim=2)[:, :, :int(w.shape[2] * self.p)]
        d1 = [i.reshape(1, -1)[0].tolist() for i in d1]
        #
        w_new = w.clone()
        w_new[d0, :, d1] = 0
        #
        d1 = [i[0].nonzero().flatten().tolist() for i in w_new]
        w_new[d0, :, d1] = Softmax(dim=1)(w_new[d0, :, d1])
        return w_new


class EntropyRegularizer(nn.Module):
    def forward(self, w):
        ent = -1.0 * (w * w.log2()).sum(axis=1)
        reg = ent.mean()
        return reg
