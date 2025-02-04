import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors3D
from miprop.descriptor.descriptor_3d.base import Descriptor3D
from miprop.descriptor.descriptor_3d.base import validate_desc_vector
from miprop.utils.logging import FailedMolecule, FailedDescriptor, FailedConformer


class RDKitDescriptor3D(Descriptor3D):
    def __init__(self, desc_name):
        super().__init__()
        self.desc_name = desc_name
        self.column_name = desc_name.replace('Calc', '')

    def _mol_to_descr(self, mol):
        bag_of_desc = []
        desc_function = getattr(Descriptors3D.rdMolDescriptors, self.desc_name)
        for conf in mol.GetConformers():
            desc_vector = np.array(desc_function(mol, confId=conf.GetId()))  # TODO implement the validate descriptor_3d functim (nan, e+287, etc)

            # desc_vector = validate_desc_vector(desc_vector)
            desc_vector = validate_desc_vector(desc_vector)

            bag_of_desc.append(desc_vector)

        return np.array(bag_of_desc)

    def transform(self, list_of_mols):
        list_of_desc = []
        for mol_id, mol in enumerate(list_of_mols):
            if isinstance(mol, (FailedMolecule, FailedConformer)):
                desc_vector = mol
            try:
                desc_vector = self._mol_to_descr(mol)
            except Exception as e:
                desc_vector = FailedDescriptor(mol)
            list_of_desc.append(desc_vector)

        return list_of_desc


class RDKitGEOM(RDKitDescriptor3D):
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
        return desc_df.values


class RDKitAUTOCORR(RDKitDescriptor3D):
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
