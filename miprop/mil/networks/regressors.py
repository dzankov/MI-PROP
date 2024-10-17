import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from miprop.mil.networks.modules.attention import AttentionNet, SelfAttentionNet, GatedAttentionNet, \
    TemperatureAttentionNet, GumbelAttentionNet, GlobalTemperatureAttentionNet
from miprop.mil.networks.modules.base import BaseRegressor
from miprop.mil.networks.modules.dynamic import DynamicPoolingNet
from miprop.mil.networks.modules.gaussian import GaussianPoolingGlobalNet
from miprop.mil.networks.modules.regular import InstanceNet, BagNet


class AttentionNetRegressor(AttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class SelfAttentionNetRegressor(SelfAttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class GatedAttentionNetRegressor(GatedAttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class TemperatureAttentionNetRegressor(TemperatureAttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class GumbelAttentionNetRegressor(GumbelAttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class GlobalTemperatureAttentionNetRegressor(GlobalTemperatureAttentionNet, BaseRegressor):
    def __init__(self):
        super().__init__()


class InstanceNetRegressor(InstanceNet, BaseRegressor):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)
        self.pool = pool

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class BagNetRegressor(BagNet, BaseRegressor):
    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class GPGlobalNetRegressor(GaussianPoolingGlobalNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, pool='lse', init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, pool=pool, init_cuda=init_cuda)


class DynamicPoolingNetRegressor(DynamicPoolingNet, BaseRegressor):
    def __init__(self, ndim=None, init_cuda=True):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x), np.asarray(y)
        x, m = add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        if isinstance(self, BaseRegressor):
            self.scaler = MinMaxScaler()
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = self.scaler.transform(y_val.reshape(-1, 1)).flatten()

        x_train, y_train, m_train = self._array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self._array_to_tensor(x_val, y_val, m_val)
        return x_train, x_val, y_train, y_val, m_train, m_val
