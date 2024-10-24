import numpy as np
from sklearn.impute import SimpleImputer


def clean_nan_descr(df_desc):
    nan_cols = list(df_desc.columns[df_desc.isnull().any(axis=0)])
    if nan_cols:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_desc[:] = imp.fit_transform(df_desc)  # TODO this is only a temporary solution, so should be revised
        return df_desc
    else:
        return df_desc
class Descriptor:
    def __init__(self):
        super().__init__()

    def calc_descriptors_for_molecules(self, list_of_mols):
        pass

    def transform(self, list_of_mols):
        return self.calc_descriptors_for_molecules(list_of_mols)


def validate_desc_vector(desc_vector):
    desc_vector = np.array(desc_vector)
    desc_vector = np.where(abs(desc_vector) <= 10 ** 35, desc_vector, np.nan)  # TODO temp solution, implement more robust one
    desc_vector = list(desc_vector)
    return desc_vector
