import os

SAVED_DATASETS = "saved_datasets"


def get_data_folder_path():
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, SAVED_DATASETS)
