from copy import deepcopy
from miprop.conformer import ConformerGenerator
from miprop.descriptor import Descriptor
from miprop.scaler import BagScaler
from miprop.mil.base_nets import BaseRegressor, BaseClassifier


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

