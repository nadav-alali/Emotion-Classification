from components.datasets.get_data import GetData
import wget
import os

from components.datasets.utils import get_data_folder_path

FIRST_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv"
SECOND_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv"
THIRD_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"

URLS = [FIRST_URL, SECOND_URL, THIRD_URL]

OUTPUT_FILE_NAME = "go_emotions"
OUTPUT_FILE_TYPE = "csv"


def _get_file_name(file_number):
    return f"{OUTPUT_FILE_NAME}_{file_number}.{OUTPUT_FILE_TYPE}"


class GoEmotionsDataset(GetData):

    def __init__(self):
        self.dataset = None

    def get_data(self) -> list:
        if self.dataset:
            return self.dataset
        self.load_data()
        return self.dataset

    def is_data_exists(self) -> bool:
        for i in range(len(URLS)):
            if not os.path.exists(_get_file_name(i)):
                return False
        return True

    def load_data(self):
        path = get_data_folder_path()
        for i, url in enumerate(URLS):
            output_file_path = os.path.join(path, _get_file_name(i))
            wget.download(url, output_file_path)
