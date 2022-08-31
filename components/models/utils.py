import random

from imblearn.over_sampling import SMOTE

from components.datasets.get_data import GetData
import numpy as np
from sklearn.model_selection import train_test_split as tts


def shuffle(data: list, labels: list):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    return zip(*zipped_data)


def train_test_split(dataset: GetData, train_percentage: float):
    p = dataset.phrases[:]
    p.extend(["" for i in range(len(dataset.data) - len(p))])  # placeholder - we didn't use the phrases in the end
    return tts(p, dataset.data, dataset.labels, train_size=train_percentage, random_state=42)


def balance_data(phrases, data, labels):
    numeric_labels = np.argmax(labels, axis=1)
    oversample = SMOTE()
    X, y = oversample.fit_resample(np.array([x[0].numpy().flatten() for x in data]), numeric_labels)
    X = X.reshape(len(X), data[0].shape[1], data[0].shape[2])
    new_y = np.zeros((y.size, y.max() + 1))
    new_y[np.arange(y.size), y] = 1
    # we didn't balance the phrases because in the end we didn't really used them
    return phrases, X, new_y
