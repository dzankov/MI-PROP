import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from copy import deepcopy
import numpy as np
import pandas as pd

from miprop.utils.logging import FailedMolecule, FailedConformer


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data


def parse_data(df):
    mols, labels = [], []
    for smi, label in zip(df[0], df[1]):
        labels.append(label)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
        else:
            mols.append(FailedMolecule(smi))
    return mols, labels


class Dataset:
    def __init__(self, file_path=None):
        super().__init__()
        self._df = load_data(file_path)
        self.molecules, self.labels = parse_data(self._df)

    def __len__(self):
        return len(self.molecules)

    def filter_failed_instances(self, failed_class):
        filtered_mols, filtered_labels = [], []
        for mol, label in zip(self.molecules, self.labels):
            if isinstance(mol, failed_class):
                continue
            filtered_mols.append(mol)
            filtered_labels.append(label)

        dataset_filtered = deepcopy(self)
        dataset_filtered.molecules = filtered_mols
        dataset_filtered.labels = filtered_labels
        return dataset_filtered

    def filter_failed_molecules(self):
        return self.filter_failed_instances(FailedMolecule)

    def filter_failed_conformers(self):
        return self.filter_failed_instances(FailedConformer)


