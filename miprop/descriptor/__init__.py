
class Descriptor:
    def __init__(self):
        super().__init__()

    def calc_descriptors_for_list_of_mols(self, list_of_mols):
        pass

    def transform(self, list_of_mols):
        return self.calc_descriptors_for_list_of_mols(list_of_mols)
