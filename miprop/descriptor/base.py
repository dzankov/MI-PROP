import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, Descriptors3D
from sklearn.impute import SimpleImputer
from molfeat.calc import FPCalculator
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs


def clean_nan_desc(df_desc):
    nan_cols = list(df_desc.columns[df_desc.isnull().any(axis=0)])
    if nan_cols:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_desc[:] = imp.fit_transform(df_desc)  # TODO this is only a temporary solution, so should be revised
        return df_desc
    else:
        return df_desc


def validate_2d_desc(mol_desc):  # TODO unify with 3D descriptors
    res = {}
    for k, v in mol_desc.items():
        if abs(v) >= 10 ** 10:
            res[k] = np.nan
        else:
            res[k] = v

    return res


def validate_desc_vector(desc_vector):
    desc_vector = np.array(desc_vector)
    desc_vector = np.where(abs(desc_vector) <= 10 ** 35, desc_vector, np.nan)  # TODO temp solution, implement more robust one
    desc_vector = list(desc_vector)
    return desc_vector


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

    def calc_descriptors_for_molecules(self, list_of_mols):
        pass

    def transform(self, list_of_mols):
        return self.calc_descriptors_for_molecules(list_of_mols)


class Descriptor2D(Descriptor):
    def __init__(self):
        super().__init__()
        self.calc = None

    def _mol_to_descr(self, mol):
        descr = self.calc(mol)
        return descr

    def calc_descriptors_for_molecules(self, list_of_mols):
        list_of_desc = []
        for mol in list_of_mols:
            mol_desc = self._mol_to_descr(mol)
            list_of_desc.append(mol_desc)
        df_desc = pd.DataFrame(list_of_desc)
        df_desc = clean_nan_desc(df_desc)
        return df_desc

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_molecules(list_of_mols)
        return df_descr


class MolFeatDescriptorFP(Descriptor2D):
    def __init__(self, method=None):
        super().__init__()
        self.calc = FPCalculator(method)

    def _mol_to_descr(self, mol):
        descr = self.calc(mol)
        return descr


class RDKitDescriptor3D(Descriptor):
    def __init__(self, desc_name):
        super().__init__()
        self.desc_name = desc_name
        self.column_name = desc_name.replace('Calc', '')

    def _mol_to_descr(self, mol):
        desc_dict = {}
        desc_function = getattr(Descriptors3D.rdMolDescriptors, self.desc_name)
        for conf in mol.GetConformers():
            desc_vector = desc_function(mol, confId=conf.GetId())  # TODO implement the validate descriptor_3d functim (nan, e+287, etc)
            desc_vector = validate_desc_vector(desc_vector)
            desc_dict[conf.GetId()] = desc_vector
        #
        columns = [f'{self.column_name}_{n}' for n in range(len(desc_vector))]
        return pd.DataFrame.from_dict(desc_dict, orient='index', columns=columns)

    def calc_descriptors_for_molecules(self, list_of_mols):
        list_of_desc = []
        for mol_id, mol in enumerate(list_of_mols):

            while not mol.GetNumConformers(): # TODO temp solution
                mol = select_closest_mol(list_of_mols, mol)

            mol_desc = self._mol_to_descr(mol)
            mol_desc = mol_desc.set_index([pd.Index([mol_id for _ in mol_desc.index])])
            list_of_desc.append(mol_desc)
        df_descr = pd.concat(list_of_desc)
        df_descr = clean_nan_desc(df_descr)
        return df_descr

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_molecules(list_of_mols)
        return df_descr


class MolFeatDescriptor3D(Descriptor): # TODO unify with Descriptor3D class
    def __init__(self):
        super().__init__()
        self.calc = None

    def _mol_to_descr(self, mol):
        desc_dict = {}
        for conf in mol.GetConformers():
            desc_vector = self.calc(mol, conformer_id=conf.GetId())  # TODO implement the validate descriptor_3d functim (nan, e+287, etc)
            desc_vector = validate_desc_vector(desc_vector)
            desc_dict[conf.GetId()] = desc_vector
        return pd.DataFrame.from_dict(desc_dict, orient='index')

    def calc_descriptors_for_molecules(self, list_of_mols):
        list_of_desc = []
        for mol_id, mol in enumerate(list_of_mols):

            while not mol.GetNumConformers():  # TODO temp solution
                mol = select_closest_mol(list_of_mols, mol)

            mol_desc = self._mol_to_descr(mol)
            mol_desc = mol_desc.set_index([pd.Index([mol_id for _ in mol_desc.index])])
            list_of_desc.append(mol_desc)
        df_descr = pd.concat(list_of_desc)
        df_descr = clean_nan_desc(df_descr)
        return df_descr

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_molecules(list_of_mols)
        return df_descr
