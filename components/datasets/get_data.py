import os.path

from components.datasets.dataset_enum import Dataset
import pickle

from components.datasets.utils import get_data_file_path, get_data_file
from components.text_handler.embedding.embedding import Embedding


class GetData:

    def __init__(self, data_type: Dataset, embedding: Embedding):
        self.data_type = data_type
        self.embedding = embedding
        self.data = []
        self.labels = []
        self.restore_data()

    def is_data_exists(self) -> bool:
        pass

    def load_data(self):
        pass

    def restore_data(self):
        if os.path.exists(get_data_file_path(self.data_type)):
            data_file = get_data_file(self.data_type, False)
            restored_data = pickle.load(data_file)
            self.data = restored_data["data"]
            self.labels = restored_data["labels"]
            data_file.close()

    def save_data(self):
        data_file = get_data_file(self.data_type, True)
        pickle.dump({"data": self.data, "labels": self.labels}, data_file)
        data_file.close()

    def get_text_label_from_label_vector(self, label_vector: list) -> str:
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        phrase = self.data[index]
        label = self.labels[index]
        return phrase, label
