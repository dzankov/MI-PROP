import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh, ReLU
from torch.nn.functional import softmax
from miprop.mil.networks.modules.base import BaseClassifier, BaseNet, MainNet


# from .modules import HopfieldPooling


class AttentionNet(BaseNet):
    def __init__(self):
        super().__init__()

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNet((input_layer_size, *hidden_layer_sizes))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        #
        input_dim = hidden_layer_sizes[-1]
        # input_dim = ndim[0]
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
        w = WeightsDropout(p=self.weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class TemperatureAttentionNet(AttentionNet):

    def __init__(self):
        super().__init__()

    def forward(self, x, m):
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / self.weight_dropout)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GlobalTemperatureAttentionNet(AttentionNet):

    def forward(self, x, m):
        temp = self.weight_dropout.to(x.device)

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / temp)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GumbelAttentionNet(AttentionNet):

    def __init__(self):
        super().__init__()

    def forward(self, x, m):

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        if self.weight_dropout == 0:
            w = Softmax(dim=2)(x_det)
        else:
            w = nn.functional.gumbel_softmax(x_det, tau=self.weight_dropout, dim=2)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)

        return w, out


class GatedAttentionNet(AttentionNet, BaseNet):
    def __init__(self):
        super().__init__()

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNet((input_layer_size, *hidden_layer_sizes))
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
        w = WeightsDropout(p=self.weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class SelfAttentionNet(BaseNet):
    def __init__(self):
        super().__init__()

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNet((input_layer_size, *hidden_layer_sizes))
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
        w = WeightsDropout(p=self.weight_dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


# class HopfieldNet(BaseNet):
#     def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
#         super().__init__(init_cuda=init_cuda)
#         self.main_net = MainNet(ndim)
#         self.estimator = Linear(ndim[-1], 1)
#         #
#         input_dim = ndim[-1]
#         self.pooling = HopfieldPooling(input_size=input_dim, hidden_size=input_dim, output_size=input_dim, num_heads=4)
#
#         if init_cuda:
#             self.main_net.cuda()
#             self.pooling.cuda()
#             self.estimator.cuda()
#
#     def forward(self, x, m):
#         x = self.main_net(x)
#
#         x = self.pooling(x)
#
#         x = x.view(-1, 1, x.shape[-1])
#
#         out = self.estimator(x)
#         if isinstance(self, BaseClassifier):
#             out = Sigmoid()(out)
#         out = out.view(-1, 1)
#         w = None
#         return w, out

# class HopfieldNetClassifier(HopfieldNet, BaseClassifier):
#     def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
#         super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)
#
#
# class HopfieldNetRegressor(HopfieldNet, BaseRegressor):
#     def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
#         super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


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


class WeightsDropout(nn.Module):
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
