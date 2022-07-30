import os

SAVED_DATASETS = "saved_datasets"


def get_data_folder_path():
    dirname = os.path.dirname(__file__)
    saved_data_path = os.path.join(dirname, SAVED_DATASETS)
    if not os.path.exists(saved_data_path):
        os.makedirs(saved_data_path)
    return saved_data_path
