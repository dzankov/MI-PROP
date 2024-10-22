import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

import numpy as np
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data


def parse_data(df):
    data = []
    for smi, label in zip(df[0], df[1]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            data.append({'mol': mol, 'label': label})
    return data


class Dataset:
    def __init__(self, file_path=None):
        super().__init__()
        self._df = load_data(file_path)
        self.data = parse_data(self._df)
        self.meta = {'num_molecules': len(self.data),
                     'mean_num_rotatable_bonds': self._calc_num_rotatable_bonds()}

    def __len__(self):
        return len(self.data)

    def get_molecules(self):
        return [i['mol'] for i in self.data]

    def get_labels(self):
        return [i['label'] for i in self.data]

    def _calc_num_rotatable_bonds(self):
        avg_rb_num = np.mean([CalcNumRotatableBonds(i['mol']) for i in self.data]).item()
        return avg_rb_num

    def _calc_num_conformers(self):
        avg_conf_num = np.mean([i['mol'].GetNumConformers() for i in self.data]).item()
        return avg_conf_num

    def _calc_conformer_diversity(self):
        pass

    def calc_conformer_stats(self):
        self.meta['mean_num_conformers'] = self._calc_num_conformers()
