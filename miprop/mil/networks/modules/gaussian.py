import torch
from torch.nn import Sequential, Linear, Sigmoid, ReLU
from miqsar.estimators.neural_nets.base_nets import BaseClassifier, BaseNet
from miprop.mil.networks.modules.regular import Pooling
from miprop.mil.networks.modules.base import MainNet


class GPNet(BaseNet):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.detector = Sequential(Linear(ndim[-1], det_ndim[0]), ReLU(), Linear(det_ndim[0], 1))
        self.estimator = Linear(ndim[-1], 1)
        self.pool = pool

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def reset_weights(self):
        self.main_net.apply(self.reset_params)
        self.detector.apply(self.reset_params)
        self.estimator.apply(self.reset_params)
        
    def forward(self, x, m):
        x = self.main_net(x)
        out = self.estimator(x)
        return x, out

    def gaussian_weighting(self, x, m, s):
        m = m.to(x.device)
        s = s.to(x.device)

        z = (x - m) / s
        w = torch.exp(-(z ** 2))
        return w

    def pooling(self, out, m):
        out = Pooling(pool=self.pool)(out, m)
        return out


class GPGlobalNet(GPNet):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, pool=pool, init_cuda=init_cuda)
        self.m = torch.nn.Parameter(torch.Tensor([0.]))
        self.s = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x, m):
        x, out = super().forward(x, m)
        x = self.detector(x)

        w = self.gaussian_weighting(x, self.m, self.s)

        out = w * out
        out = self.pooling(out, m)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        w = w.reshape(w.shape[0], w.shape[2], w.shape[1])
        return w, out


