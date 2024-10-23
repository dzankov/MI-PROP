from rdkit import Chem
from rdkit.Chem import AllChem

from miprop.conformer_generation.base import ConformerGenerator


class RDKitConformerGenerator(ConformerGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_molecule(self, mol):
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        mol = self._prepare_molecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, maxAttempts=700, randomSeed=42)
        return mol
