import pandas as pd
from rdkit.Chem import Descriptors3D
from miprop.descriptor_calculation.base import RDKitDescriptor3D


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
