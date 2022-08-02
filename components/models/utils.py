import random

import torch

from components.datasets.get_data import GetData

LOSS = "loss"
TRAIN = "train"
TEST = "test"
ACC = "accuracy"
ROC = "roc accuracy"
F1 = "F1 accuracy"


def shuffle(data: list, labels: list):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    return zip(*zipped_data)


def train_test_split(dataset: GetData, train_percentage: float):
    train_size = round(len(dataset) * train_percentage)
    train_dataset_data, train_dataset_labels = dataset[: train_size]
    test_dataset_data, test_dataset_labels = dataset[train_size:]
    return train_dataset_data, train_dataset_labels, test_dataset_data, test_dataset_labels


def feed_model(model, hidden_size, X, y, data_size, batch_size, criterion, optimizer, accuracy, roc_accuracy,
               F1_accuracy, measurements, mode=TRAIN):
    curr_measurement_sum = {
        LOSS: 0.0,
        ACC: 0.0,
        ROC: 0.0,
        F1: 0.0
    }
    amount = 0.0
    # train the model
    for index in range(0, data_size, batch_size):
        amount += 1
        phrases = torch.stack(X[index: index + batch_size])
        labels = torch.tensor(y[index: index + batch_size], dtype=float)

        curr_batch_size = len(labels)
        hidden_state = torch.zeros(curr_batch_size, hidden_size)
        for i in range(curr_batch_size):
            output, hidden_state = model(phrases[:, i, :], hidden_state)

        loss = criterion(output, labels)

        # measure
        int_labels = labels.int()
        curr_measurement_sum[LOSS] += loss.item()
        curr_measurement_sum[ACC] += accuracy(output, int_labels)
        curr_measurement_sum[ROC] += roc_accuracy(output, int_labels)
        curr_measurement_sum[F1] += F1_accuracy(output, int_labels)

        # train
        if mode == TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for measurement in measurements.keys():
        measurements[measurement][mode].append(curr_measurement_sum[measurement] / amount)
