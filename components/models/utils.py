import random

from components.datasets.get_data import GetData


def shuffle(data: list, labels: list):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    return zip(*zipped_data)


def train_test_split(dataset: GetData, train_percentage: float):
    train_size = round(len(dataset) * train_percentage)
    train_dataset_data, train_dataset_labels = dataset[: train_size]
    test_dataset_data, test_dataset_labels = dataset[train_size:]
    return train_dataset_data, train_dataset_labels, test_dataset_data, test_dataset_labels
