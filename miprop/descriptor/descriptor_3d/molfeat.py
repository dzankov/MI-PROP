from rdkit import Chem
import numpy as np
import pandas as pd
from molfeat.calc import Pharmacophore3D
from molfeat.calc import USRDescriptors
from molfeat.calc import ElectroShapeDescriptors
from miprop.descriptor.base import Descriptor3D, validate_desc_vector


class MolFeatDescriptor3D(Descriptor3D):  # TODO unify with Descriptor3D class
    def __init__(self):
        super().__init__()
        self.transformer = None

    def _mol_to_descr(self, mol):
        bag_of_desc = []
        for conf in mol.GetConformers():
            desc_vector = self.transformer(mol, conformer_id=conf.GetId())  # TODO implement the validate descriptor_3d functim (nan, e+287, etc)

            desc_vector = validate_desc_vector(desc_vector)
            bag_of_desc.append(desc_vector)

        return np.array(bag_of_desc)

    def transform(self, list_of_mols):
        list_of_desc = []
        for mol_id, mol in enumerate(list_of_mols):
            try:
                desc_vector = self._mol_to_descr(mol)
            except Exception as e:
                print(Chem.MolToSmiles(mol), e)
            list_of_desc.append(desc_vector)

        return list_of_desc


class MolFeatPharmacophore(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.transformer = Pharmacophore3D()


class MolFeatUSRD(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.transformer = USRDescriptors()


class MolFeatElectroShape(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.transformer = ElectroShapeDescriptors()
