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
    return tts(dataset.phrases, dataset.data, dataset.labels, train_size=train_percentage, random_state=42)


def balance_data(phrases, data, labels):
    numeric_labels = np.argmax(labels, axis=1)
    oversample = SMOTE()
    _X, y = oversample.fit_resample(np.arange(len(numeric_labels)).reshape(-1, 1), numeric_labels)
    X = np.array([data[i[0]] for i in _X])
    Z = [phrases[i[0]] for i in _X]
    new_y = np.zeros((y.size, y.max() + 1))
    new_y[np.arange(y.size), y] = 1
    return Z, X, new_y
