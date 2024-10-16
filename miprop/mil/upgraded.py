import numpy as np
from misvm import MISVM, miSVM, NSK, STK, MissSVM, MICA, sMIL, stMIL, sbMIL


class BaseEstimator:

    def __init__(self):
        pass

    def predict(self, bags):
        predictions = self.super().predict(bags)
        return np.sign(predictions)


class MISVM(BaseEstimator, MISVM):
    pass

class miSVM(BaseEstimator, miSVM):
    pass

class NSK(BaseEstimator, NSK):
    pass

class STK(BaseEstimator, STK):
    pass

class MissSVM(BaseEstimator, MissSVM):
    pass

class MICA(BaseEstimator, MICA):
    pass

class sMIL(BaseEstimator, sMIL):
    pass

class stMIL(BaseEstimator, stMIL):
    pass

class sbMIL(BaseEstimator, sbMIL):
    pass