from torch.nn import Sequential, Linear, Softmax, Sigmoid
from miprop.mil.network.module.base import BaseNetwork, BaseClassifier, MainNetwork, Pooling


class BagNetwork(BaseNetwork):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):
        self.main_net = MainNetwork((input_layer_size, *hidden_layer_sizes))
        self.pooling = Pooling(self.pool)
        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        out = self.main_net(x)
        out = self.pooling(out, m)
        out = self.estimator(out)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        return None, out


class InstanceNetwork(BaseNetwork):

    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):
        self.main_net = Sequential(MainNetwork((input_layer_size, *hidden_layer_sizes)), Linear(hidden_layer_sizes[-1], 1))
        self.pooling = Pooling(self.pool)

        if self.init_cuda:
            self.main_net.cuda()

    def forward(self, x, m):
        out = self.main_net(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        w = Softmax(dim=1)(m * out)
        w = w.view(w.shape[0], w.shape[-1], w.shape[1])
        out = self.pooling(out, m)
        return w, out


