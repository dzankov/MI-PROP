import numpy as np
from sklearn.impute import SimpleImputer
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from miprop.utils.dataset import Dataset

# nan descr
# inf descr


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


def validate_2d_desc(mol_desc):  # TODO unify with 3D descriptors
    res = {}
    for k, v in mol_desc.items():
        if abs(v) >= 10 ** 10:
            res[k] = np.nan
        else:
            res[k] = v

    return res

def select_closest_mol(list_of_mols, mol):
    tmp = []
    fp_gen = GetMorganGenerator(radius=3, fpSize=2048)
    for mol_i in list_of_mols:
        fp_gen.GetFingerprint(mol_i)
        sim = DataStructs.TanimotoSimilarity(fp_gen.GetFingerprint(mol_i), fp_gen.GetFingerprint(mol))
        tmp.append((mol_i, sim))
    tmp = sorted(tmp, key=lambda x: x[1])
    return tmp[0][0]


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


class Descriptor2D(Descriptor):
    def __init__(self):
        super().__init__()


class Descriptor3D(Descriptor):
    def __init__(self):
        super().__init__()



