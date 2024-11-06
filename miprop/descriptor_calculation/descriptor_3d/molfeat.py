from molfeat.calc import Pharmacophore3D
from molfeat.calc import USRDescriptors
from molfeat.calc import ElectroShapeDescriptors
from ..base import Descriptor, validate_desc_vector, clean_nan_descr
import pandas as pd


class MolFeatDescriptor3D(Descriptor):  # TODO unify with Descriptor3D class
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
            mol_desc = self._mol_to_descr(mol)
            mol_desc = mol_desc.set_index([pd.Index([mol_id for _ in mol_desc.index])])
            list_of_desc.append(mol_desc)
        df_descr = pd.concat(list_of_desc)
        df_descr = clean_nan_descr(df_descr)
        return df_descr

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_molecules(list_of_mols)
        return df_descr


class PharmacophoreDescriptor3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = Pharmacophore3D()


class USRDescriptor3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = USRDescriptors()


class ElectroShapeDescriptors3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = ElectroShapeDescriptors()
