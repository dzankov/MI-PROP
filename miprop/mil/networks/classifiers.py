from miprop.mil.networks.modules.attention import AttentionNetwork, SelfAttentionNetwork, GatedAttentionNetwork, \
    TemperatureAttentionNetwork, GlobalTemperatureAttentionNetwork
from miprop.mil.networks.modules.base import BaseClassifier
from miprop.mil.networks.modules.dynamic import DynamicPoolingNetwork, MarginLoss
from miprop.mil.networks.modules.gaussian import GaussianPoolingNetwork
from miprop.mil.networks.modules.hopfield import HopfieldNetwork
from miprop.mil.networks.modules.regular import BagNetwork, InstanceNetwork


class AttentionNetClassifier(AttentionNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class SelfAttentionNetClassifier(SelfAttentionNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GatedAttentionNetClassifier(GatedAttentionNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class TemperatureAttentionNetworkClassifier(TemperatureAttentionNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GlobalTemperatureAttentionNetworkClassifier(GlobalTemperatureAttentionNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class BagNetClassifier(BagNetwork, BaseClassifier):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class InstanceNetClassifier(InstanceNetwork, BaseClassifier):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class GPNetworkClassifier(GaussianPoolingNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, pool=pool, init_cuda=init_cuda)


class DynamicPoolingNetClassifier(DynamicPoolingNetwork, BaseClassifier):
    def __init__(self, ndim=None, init_cuda=True):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

    def loss(self, y_pred, y_true):
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true.reshape(-1, 1))
        return loss


class HopfieldNetworkClassifier(HopfieldNetwork, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)
