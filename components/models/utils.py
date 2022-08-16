import random

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from components.datasets.get_data import GetData
import numpy as np
from sklearn.model_selection import train_test_split as tts


def shuffle(data: list, labels: list):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    return zip(*zipped_data)


def train_test_split(dataset: GetData, train_percentage: float):
    # train_size = round(len(dataset) * train_percentage)
    # train_dataset_data, train_dataset_labels = dataset[: train_size]
    # test_dataset_data, test_dataset_labels = dataset[train_size:]
    # return train_dataset_data, train_dataset_labels, test_dataset_data, test_dataset_labels
    return tts(dataset.data, dataset.labels, train_size=train_percentage, random_state=42)


def balance_data(data, labels):
    numeric_labels = np.argmax(labels, axis=1)
    oversample = SMOTE()
    X, y = oversample.fit_resample(np.array([x[0].numpy().flatten() for x in data]), numeric_labels)
    X = X.reshape(len(X), data[0].shape[1], data[0].shape[2])
    new_y = np.zeros((y.size, y.max() + 1))
    new_y[np.arange(y.size), y] = 1
    return X, new_y
