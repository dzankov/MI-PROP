import pandas as pd
from rdkit.Chem import Descriptors3D
import numpy as np
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from . import Descriptor


class RDKitDescriptor(Descriptor):
    def __init__(self, desc_name):
        super().__init__()
        self.desc_name = desc_name
        self.column_name = desc_name.replace('Calc', '')

    def _mol_to_descr(self, mol):
        desc_dict = {}
        for conf in mol.GetConformers():
            desc_vector = getattr(Descriptors3D.rdMolDescriptors, self.desc_name)(mol, confId=conf.GetId())
            desc_dict[conf.GetId()] = desc_vector
        #
        columns = [f'{self.column_name}_{n}' for n in range(len(desc_vector))]
        return pd.DataFrame.from_dict(desc_dict, orient='index', columns=columns)

    def calc_descriptors_for_list_of_mols(self, list_of_mols):
        list_of_descr = []
        # for mol_id, mol in enumerate(list_of_mols):
        for mol_id, mol in tqdm(enumerate(list_of_mols),
                                total=len(list_of_mols),
                                desc=f"{self.__class__.__name__} descriptor calculation: ",
                                bar_format="{desc}{n}/{total} [{elapsed}]",
                                ):

            mol_descr = self._mol_to_descr(mol)
            mol_descr = mol_descr.set_index([pd.Index([mol_id for _ in mol_descr.index])])
            list_of_descr.append(mol_descr)
        #
        df_descr = pd.concat(list_of_descr)
        #
        nan_cols = list(df_descr.columns[df_descr.isnull().any(axis=0)])
        if nan_cols:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            df_descr[:] = imp.fit_transform(df_descr) # TODO this is only a temporary solution, so should be revised

        return df_descr

    def calc_descriptors_for_dataset(self, dataset):
        list_of_mols = dataset.get_molecules()
        df_descr = self.calc_descriptors_for_list_of_mols(list_of_mols)
        return df_descr



class RDKitGENERAL(RDKitDescriptor):
    def __init__(self):
        super().__init__('RDKitGENERAL')

        self.desc_list = ['CalcAsphericity',
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
            column_name = desc_name.replace('Calc', '')
            for conf in mol.GetConformers():
                desc_value = getattr(Descriptors3D.rdMolDescriptors, desc_name)(mol, confId=conf.GetId())
                desc_df.loc[conf.GetId(), column_name] = desc_value
        return desc_df


class RDKitAUTOCORR3D(RDKitDescriptor):
    def __init__(self):
        super().__init__('CalcAUTOCORR3D')


class RDKitRDF(RDKitDescriptor):
    def __init__(self):
        super().__init__('CalcRDF')


class RDKitMORSE(RDKitDescriptor):
    def __init__(self):
        super().__init__('CalcMORSE')


class RDKitWHIM(RDKitDescriptor):
    def __init__(self):
        super().__init__('CalcWHIM')


class RDKitGETAWAY(RDKitDescriptor):
    def __init__(self):
        super().__init__('CalcGETAWAY')
