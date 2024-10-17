from miprop.mil.networks.modules.attention import AttentionNet, SelfAttentionNet, GatedAttentionNet, \
    TemperatureAttentionNet, GlobalTemperatureAttentionNet
from miprop.mil.networks.modules.base import BaseClassifier
from miprop.mil.networks.modules.dynamic import DynamicPoolingNet, MarginLoss
from miprop.mil.networks.modules.gaussian import GaussianPoolingGlobalNet
from miprop.mil.networks.modules.regular import BagNet, InstanceNet


class AttentionNetClassifier(AttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class SelfAttentionNetClassifier(SelfAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GatedAttentionNetClassifier(GatedAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class TemperatureAttentionNetClassifier(TemperatureAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GlobalTemperatureAttentionNetClassifier(GlobalTemperatureAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class BagNetClassifier(BagNet, BaseClassifier):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class InstanceNetClassifier(InstanceNet, BaseClassifier):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class GPGlobalNetClassifier(GaussianPoolingGlobalNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, pool=pool, init_cuda=init_cuda)


class DynamicPoolingNetClassifier(DynamicPoolingNet, BaseClassifier):
    def __init__(self, ndim=None, init_cuda=True):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

    def loss(self, y_pred, y_true):
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true.reshape(-1, 1))
        return loss
