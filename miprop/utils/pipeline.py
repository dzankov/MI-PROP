from miprop.conformer_generation import ConformerGenerator
from miprop.descriptor_3d.base import Descriptor
from miprop.utils.scaler import BagScaler
from miprop.mil.network.module.base import BaseRegressor, BaseClassifier


class Pipeline:
    def __init__(self, list_of_transformers=None):
        super().__init__()
        self.list_of_transformers = list_of_transformers

    def fit(self, x, y):
        for transformer in self.list_of_transformers:
            if isinstance(transformer, ConformerGenerator):
                x = transformer.transform(x)
            elif isinstance(transformer, Descriptor):
                x = transformer.transform(x)
            elif isinstance(transformer, BagScaler):
                transformer.fit(x)
                x = transformer.transform(x)
            elif isinstance(transformer, (BaseRegressor, BaseClassifier)):
                transformer.fit(x, y)

    def predict(self, x):
        for transformer in self.list_of_transformers:
            if isinstance(transformer, BagScaler):
                x = transformer.transform(x)
            elif isinstance(transformer, (BaseRegressor, BaseClassifier)):
                y_pred = transformer.predict(x)
            else:
                x = transformer.transform(x)

        return y_pred

