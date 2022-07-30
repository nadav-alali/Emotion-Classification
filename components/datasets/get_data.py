
class GetData:

    def __init__(self):
        self.data = []
        self.labels = []

    def is_data_exists(self) -> bool:
        pass

    def load_data(self):
        pass

    def __len__(self):
        len(self.data)

    def __getitem__(self, index):
        phrase = self.data[index]
        label = self.labels[index]
        return phrase, label

