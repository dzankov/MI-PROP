import sys

from openbabel import pybel, openbabel
from rdkit import Chem

from miprop.conformer_generation.base import ConformerGenerator


class BabelConformerGenerator(ConformerGenerator):
    def __init__(self, **kwargs):
        super().__init__( *kwargs)

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
