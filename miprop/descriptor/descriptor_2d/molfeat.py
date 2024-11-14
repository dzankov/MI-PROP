from ..base import Descriptor2D, MolFeatFingerprint2D
from molfeat.calc import RDKitDescriptors2D
from molfeat.calc import Pharmacophore2D
from molfeat.calc import MordredDescriptors
from molfeat.calc import CATS
from molfeat.calc import ScaffoldKeyCalculator


class AtomPair(MolFeatFingerprint2D): #TODO add columns name
    def __init__(self):
        super().__init__("atompair")


class AtomPairCount(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("atompair-count")


class Avalon(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("avalon")


# class AvalonCount(MolFeatDescriptorFP):
#     def __init__(self):
#         super().__init__("avalon-count")


class ECFP(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("ecfp")


class ECFPCount(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("ecfp-count")


class ERG(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("erg")


class Estate(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("estate")


class FCFP(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("fcfp")


class FCFPCount(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("fcfp-count")


class Layered(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("layered")


class MACCS(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("maccs")


class Pattern(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("pattern")


class RDKitFP(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("rdkit")


class RDKitFPCount(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("rdkit-count")


class SecFP(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("secfp")


class Topological(MolFeatFingerprint2D):
    def __init__(self):
        super().__init__("topological")


class TopologicalCount(MolFeatFingerprint2D):
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




