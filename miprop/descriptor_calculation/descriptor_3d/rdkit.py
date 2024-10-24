import pandas as pd
from rdkit.Chem import Descriptors3D
import numpy as np
from sklearn.impute import SimpleImputer
from miprop.descriptor_calculation.base import Descriptor, clean_nan_descr, validate_desc_vector


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


class RDKitGENERAL3D(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('RDKitGENERAL')

        self.desc_list = ['CalcAsphericity', # TODO replace from rdkit directly
                          'CalcEccentricity',
                          'CalcInertialShapeFactor',
                          'CalcNPR1',
                          'CalcNPR2',
                          'CalcPMI1',
                          'CalcPMI2',
                          'CalcPMI3',
                          'CalcRadiusOfGyration',
                          'CalcSpherocityIndex',
                          'CalcPBF']

    def _mol_to_descr(self, mol):
        desc_df = pd.DataFrame()
        for desc_name in self.desc_list:
            desc_function = getattr(Descriptors3D.rdMolDescriptors, desc_name)
            column_name = desc_name.replace('Calc', '')
            for conf in mol.GetConformers():
                desc_value = desc_function(mol, confId=conf.GetId())
                desc_df.loc[conf.GetId(), column_name] = desc_value
        return desc_df


class RDKitAUTOCORR3D(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcAUTOCORR3D')


class RDKitRDF(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcRDF')


class RDKitMORSE(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcMORSE')


class RDKitWHIM(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcWHIM')


class RDKitGETAWAY(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcGETAWAY')
