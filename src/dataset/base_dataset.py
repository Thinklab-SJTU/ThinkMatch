class BaseDataset:
    def __init__(self):
        pass

    def get_pair(self, cls, shuffle):
        raise NotImplementedError
