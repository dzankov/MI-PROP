import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from miprop.mil.network.module.attention import AttentionNetwork, SelfAttentionNetwork, GatedAttentionNetwork, \
    TemperatureAttentionNetwork, GumbelAttentionNetwork, GlobalTemperatureAttentionNetwork
from miprop.mil.network.module.base import BaseRegressor, add_padding
from miprop.mil.network.module.dynamic import DynamicPoolingNetwork
from miprop.mil.network.module.gaussian import GaussianPoolingNetwork
from miprop.mil.network.module.hopfield import HopfieldNetwork
from miprop.mil.network.module.classic import InstanceNetwork, BagNetwork


class AttentionNetworkRegressor(AttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SelfAttentionNetworkRegressor(SelfAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GatedAttentionNetworkRegressor(GatedAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TemperatureAttentionNetworkRegressor(TemperatureAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GumbelAttentionNetworkRegressor(GumbelAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GlobalTemperatureAttentionNetworkRegressor(GlobalTemperatureAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InstanceNetworkRegressor(InstanceNetwork, BaseRegressor):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)
        self.pool = pool


class BagNetworkRegressor(BagNetwork, BaseRegressor):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)


class GaussianPoolingNetworkRegressor(GaussianPoolingNetwork, BaseRegressor):
    def __init__(self, pool='lse', **kwargs):
        super().__init__(pool=pool, **kwargs)


class DynamicPoolingNetworkRegressor(DynamicPoolingNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x, dtype="object"), np.asarray(y, dtype="object")
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


class HopfieldNetworkRegressor(HopfieldNetwork, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)