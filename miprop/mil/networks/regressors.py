from miprop.mil.networks.modules.attention import AttentionNet, SelfAttentionNet, GatedAttentionNet, \
    TemperatureAttentionNet, GumbelAttentionNet, GlobalTemperatureAttentionNet
from miprop.mil.networks.modules.base import BaseRegressor
from miprop.mil.networks.modules.gaussian import GPGlobalNet
from miprop.mil.networks.modules.regular import InstanceNet, BagNet


class AttentionNetRegressor(AttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class SelfAttentionNetRegressor(SelfAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GatedAttentionNetRegressor(GatedAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class TemperatureAttentionNetRegressor(TemperatureAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GumbelAttentionNetRegressor(GumbelAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GlobalTemperatureAttentionNetRegressor(GlobalTemperatureAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class InstanceNetRegressor(InstanceNet, BaseRegressor):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)
        self.pool = pool

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class BagNetRegressor(BagNet, BaseRegressor):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class GPGlobalNetRegressor(GPGlobalNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, pool=pool, init_cuda=init_cuda)
