from molfeat.calc import Pharmacophore3D
from molfeat.calc import USRDescriptors
from molfeat.calc import ElectroShapeDescriptors
from ..base import MolFeatDescriptor3D


class PharmacophoreDescriptor3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = Pharmacophore3D()


class USRDescriptor3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = USRDescriptors()


class ElectroShapeDescriptors3D(MolFeatDescriptor3D):
    def __init__(self):
        super().__init__()
        self.calc = ElectroShapeDescriptors()
