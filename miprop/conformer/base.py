from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from joblib import Parallel, delayed

RDLogger.DisableLog('rdApp.*')


class ConformerGenerator:
    def __init__(self, num_conf=10, e_thresh=None, num_cpu=1):
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh
        self.num_cpu = num_cpu

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
        if not mol.GetNumConformers():
            return mol
        if self.e_thresh:
            mol = filter_by_energy(mol, self.e_thresh)
        return mol

    def generate_conformers_for_molecules(self, list_of_mols):

        futures = Parallel(n_jobs=self.num_cpu)(delayed(self._generate_conformers)(mol) for mol in list_of_mols)
        list_of_mols_with_confs = []
        for mol in futures:
            if not mol.GetNumConformers():
                print(Chem.MolToSmiles(mol), ' Conformer generation failed')
            list_of_mols_with_confs.append(mol)
        return list_of_mols_with_confs

    def generate_conformers_for_dataset(self, dataset):

        futures = Parallel(n_jobs=self.num_cpu)(delayed(self._generate_conformers)(record['mol']) for record in dataset.data)
        list_of_records = []
        for n, record in enumerate(dataset.data):
            mol = futures[n]
            if not mol.GetNumConformers():
                print(Chem.MolToSmiles(mol), ' Conformer generation failed')
            record['mol'] = mol
            list_of_records.append(record)
        dataset.data = list_of_records
        dataset.calc_conformer_stats()
        return dataset

    def transform(self, list_of_mols):
        return self.generate_conformers_for_molecules(list_of_mols)


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
