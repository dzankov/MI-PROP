from rdkit import Chem
from rdkit.Chem import AllChem


class ConformerGenerator:
    def __init__(self, num_conf=10):
        super().__init__()

        self.num_conf = num_conf

    def _prepare_molecule(self, mol):
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        pass

    def _optimize_conformers(self, mol):
        pass

    def generate_conformers(self, mol):
        pass


class ConformerFilter:
    def __init__(self):
        super().__init__()

    def filter_by_energy(self, mol, e_threshold=0):

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
            if conf_energy - min_energy >= e_threshold:
                mol.RemoveConformer(conf_id)

        return mol

    def filter_by_rmsd(self, mol, rmsd_threshold=2):
        pass


class RDKitConformerGenerator(ConformerGenerator):
    def __init__(self, num_conf=10):
        super().__init__()

        self.num_conf = num_conf

    def _prepare_molecule(self, mol):
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, maxAttempts=700, randomSeed=42)
        return mol

    def _optimize_conformers(self, mol):
        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def generate_conformers(self, mol):
        mol = self._prepare_molecule(mol)
        mol = self._embedd_conformers(mol)
        mol = self._optimize_conformers(mol)
        return mol


class BabelConformerGenerator(ConformerGenerator):
    pass

