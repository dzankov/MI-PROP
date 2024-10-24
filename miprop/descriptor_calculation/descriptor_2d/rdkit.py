from rdkit.Chem import AllChem
import pandas as pd
from rdkit.Chem import Descriptors

from miprop.descriptor_calculation.base import Descriptor, clean_nan_descr, validate_desc_vector


class RDKitDescriptor2D(Descriptor):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        return

    def calc_descriptors_for_molecules(self, list_of_mols):
        list_of_desc = []
        for mol in list_of_mols:
            mol_desc = self._mol_to_descr(mol)
            list_of_desc.append(mol_desc)
        df_desc = pd.DataFrame(list_of_desc)
        df_desc = clean_nan_descr(df_desc)
        return df_desc

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_molecules(list_of_mols)
        return df_descr


class RDKitGENERAL2D(RDKitDescriptor2D):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        desc_dict = {}
        for desc_name, desc_function in Descriptors._descList:
            desc_value = desc_function(mol)
            desc_dict[desc_name] = desc_value
        return desc_dict


class RDKitFingerprint(RDKitDescriptor2D):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        fpgen = AllChem.GetRDKitFPGenerator(maxPath=2, fpSize=1024)
        descr = fpgen.GetFingerprintAsNumPy(mol)
        descr = {n:d for n, d in enumerate(descr)}
        return descr


class RDKitAtomPair(RDKitDescriptor2D):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        fpgen = AllChem.GetAtomPairGenerator()
        descr = fpgen.GetFingerprintAsNumPy(mol)
        descr = {n:d for n, d in enumerate(descr)}
        return descr


class RDKitTopologicalTorsion(RDKitDescriptor2D):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        fpgen = AllChem.GetTopologicalTorsionGenerator()
        descr = fpgen.GetFingerprintAsNumPy(mol)
        descr = {n:d for n, d in enumerate(descr)}
        return descr


class RDKitMorgan(RDKitDescriptor2D):
    def __init__(self):
        super().__init__()

    def _mol_to_descr(self, mol):
        fpgen = AllChem.GetMorganGenerator(radius=2)
        descr = fpgen.GetFingerprintAsNumPy(mol)
        descr = {n:d for n, d in enumerate(descr)}
        return descr