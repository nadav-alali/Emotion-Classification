from components.datasets.get_data import GetData
import wget

from components.datasets.utils import get_data_folder_path

FIRST_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv"
SECOND_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv"
THIRD_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"

URLS = [FIRST_URL, SECOND_URL, THIRD_URL]


class GoEmotionsDataset(GetData):

    def __init__(self):
        self.dataset = None

    def get_data(self) -> list:
        if self.is_data_exists():
            return self.load_data()
        dataset = []  # todo - import data
        self.save_data(dataset)
        return dataset

    def is_data_exists(self) -> bool:
        pass

    def load_data(self) -> list:
        if self.dataset:
            return self.dataset
        path = get_data_folder_path()
        for url in URLS:
            wget.download(url, path)

    def save_data(self, dataset):
        pass
