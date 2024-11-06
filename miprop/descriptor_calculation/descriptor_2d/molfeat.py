from ..base import Descriptor2D, MolFeatDescriptorFP
from molfeat.calc import RDKitDescriptors2D
from molfeat.calc import Pharmacophore2D
from molfeat.calc import MordredDescriptors
from molfeat.calc import CATS
from molfeat.calc import ScaffoldKeyCalculator


class AtomPair(MolFeatDescriptorFP): #TODO add columns name
    def __init__(self):
        super().__init__("atompair")


class AtomPairCount(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("atompair-count")


class Avalon(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("avalon")


# class AvalonCount(MolFeatDescriptorFP):
#     def __init__(self):
#         super().__init__("avalon-count")


class ECFP(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("ecfp")


class ECFPCount(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("ecfp-count")


class ERG(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("erg")


class Estate(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("estate")


class FCFP(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("fcfp")


class FCFPCount(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("fcfp-count")


class Layered(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("layered")


class MACCS(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("maccs")


class Pattern(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("pattern")


class RDKitFP(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("rdkit")


class RDKitFPCount(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("rdkit-count")


class SecFP(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("secfp")


class Topological(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("topological")


class TopologicalCount(MolFeatDescriptorFP):
    def __init__(self):
        super().__init__("topological-count")


class RDKitPhysChem(Descriptor2D):
    def __init__(self):
        super().__init__()
        self.calc = RDKitDescriptors2D(replace_nan=True)


class Pharmacophore(Descriptor2D):
    def __init__(self):
        super().__init__()
        self.calc = Pharmacophore2D(replace_nan=True)


class Mordred(Descriptor2D):
    def __init__(self):
        super().__init__()
        self.calc = MordredDescriptors(replace_nan=True)


class ChemicallyAdvancedTemplateSearch(Descriptor2D):
    def __init__(self):
        super().__init__()
        self.calc = CATS()


class ScaffoldKeyDescriptor(Descriptor2D):
    def __init__(self):
        super().__init__()
        self.calc = ScaffoldKeyCalculator()




