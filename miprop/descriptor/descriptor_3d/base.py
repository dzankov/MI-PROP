import numpy as np
from miprop.utils.dataset import Dataset


def validate_desc_vector(desc_vector):

    # nan values
    if np.isnan(desc_vector).sum() > 0:
        imp = np.mean(desc_vector[~np.isnan(desc_vector)])
        desc_vector = np.where(np.isnan(desc_vector), imp, desc_vector)  # TODO this is only a temporary solution, so should be revised
    # extreme dsc values
    if (abs(desc_vector) >= 10 ** 25).sum() > 0:
        imp = np.mean(desc_vector[abs(desc_vector) <= 10 ** 25])
        desc_vector = np.where(abs(desc_vector) <= 10 ** 25, desc_vector, imp)
    return desc_vector


class Descriptor:
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        pass

    def transform(self, list_of_mols):

        if isinstance(list_of_mols, Dataset):
            list_of_mols = list_of_mols.molecules

        list_of_desc = []
        for mol in list_of_mols:
            x = self._mol_to_descr(mol)
            list_of_desc.append(x)

        # df_desc = pd.DataFrame(list_of_desc)
        # df_desc = clean_nan_desc(df_desc)
        # return df_desc

        return list_of_desc


class Descriptor3D(Descriptor):
    def __init__(self):
        super().__init__()



