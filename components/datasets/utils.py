import os

from components.datasets.dataset_enum import Dataset

SAVED_DATASETS = "saved_datasets"


def get_data_folder_path():
    dirname = os.path.dirname(__file__)
    saved_data_path = os.path.join(dirname, SAVED_DATASETS)
    if not os.path.exists(saved_data_path):
        os.makedirs(saved_data_path)
    return saved_data_path


def get_data_file_path(data_type: Dataset):
    return os.path.join(get_data_folder_path(), f"{data_type}")


def get_data_file(data_type: Dataset, writeable: bool):
    open_mode = 'ab' if writeable else 'rb'
    data_file = open(get_data_file_path(data_type), open_mode)
    return data_file
