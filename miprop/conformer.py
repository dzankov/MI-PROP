from rdkit import Chem
from rdkit.Chem import AllChem
import os, time
import sys
import gzip
import argparse
import pickle
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from openbabel import pybel, openbabel

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class ConformerGenerator:
    def __init__(self, num_conf=10, e_thresh=None):
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh

    def _prepare_molecule(self, mol):
        pass

    def _embedd_conformers(self, mol):
        pass

    def _optimize_conformers(self, mol):
        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def _generate_conformers(self, mol):
        mol = self._embedd_conformers(mol)
        mol = self._optimize_conformers(mol)
        if self.e_thresh:
            mol = filter_by_energy(mol, self.e_thresh)
        return mol

    def generate_conformers_for_list_of_mols(self, list_of_mols):
        list_of_mols_with_confs = []
        for mol in list_of_mols:
            mol = self._generate_conformers(mol)
            list_of_mols_with_confs.append(mol)
        return list_of_mols_with_confs

    def transform(self, list_of_mols):
        return self.generate_conformers_for_list_of_mols(list_of_mols)


class RDKitConformerGenerator(ConformerGenerator):
    def __init__(self, num_conf=10, e_thresh=None):
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh

    def _prepare_molecule(self, mol):
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        mol = self._prepare_molecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, maxAttempts=700, randomSeed=42)
        return mol


class BabelConformerGenerator(ConformerGenerator):
    def __init__(self, num_conf=10):
        super().__init__()

        self.num_conf = num_conf

    def _prepare_molecule(self, mol):
        mol_rdkit = Chem.AddHs(mol)
        mol_babel = pybel.readstring('mol', Chem.MolToMolBlock(mol)).OBMol  # convert mol from RDKit to OB
        mol_babel.AddHydrogens()
        return mol_rdkit, mol_babel

    def _embedd_conformers(self, mol):
        mol_rdkit, mol_babel = self._prepare_molecule(mol)
        #
        ff = pybel._forcefields["mmff94"]
        success = ff.Setup(mol_babel)
        if not success:
            ff = pybel._forcefields["uff"]
            success = ff.Setup(mol_babel)
            if not success:
                sys.exit("Cannot set up Open Babel force field")

        ff.DiverseConfGen(0, 100000, 100, False)  # rmsd, nconf_tries, energy, verbose
        ff.GetConformers(mol_babel)
        ff.ConjugateGradients(100, 1.0e-3)
        #
        obconversion = openbabel.OBConversion()
        obconversion.SetOutFormat('mol')
        #
        conf_mol_blocks = []
        for conf_num in range(max(0, mol_babel.NumConformers() - self.num_conf), mol_babel.NumConformers()):
            mol_babel.SetConformer(conf_num)
            conf_mol_blocks.append(obconversion.WriteString(mol_babel))

        rdkit_confs = []
        for i in conf_mol_blocks:
            conf_rdkit = Chem.MolFromMolBlock(i, removeHs=False)
            rdkit_confs.append(conf_rdkit)
        #
        for conf in rdkit_confs:
            mol_rdkit.AddConformer(conf.GetConformer())

        return mol_rdkit


def filter_by_energy(mol, e_thresh=1):
    conf_energy_list = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            continue
        conf_energy_list.append((conf.GetId(), ff.CalcEnergy()))
    conf_energy_list = sorted(conf_energy_list, key=lambda x: x[1])

    # filter conformers
    min_energy = conf_energy_list[0][1]
    for conf_id, conf_energy in conf_energy_list[1:]:
        if conf_energy - min_energy >= e_thresh:
            mol.RemoveConformer(conf_id)

    return mol


def filter_by_rmsd(mol, rmsd_thresh=2):
    pass