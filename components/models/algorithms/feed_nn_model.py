import numpy as np
import torch

LOSS = "loss"
TRAIN = "train"
TEST = "test"
ACC = "accuracy"
AUROC = "area under ROC curve"
F1 = "F1 accuracy"
PRECISION = "precision"


def _update_curr_measurement_sum(output, labels, loss, accuracy, area_under_roc, f1_score, precision,
                                 curr_measurement_sum):
    curr_measurement_sum[LOSS] += loss
    curr_measurement_sum[ACC] += accuracy(output, labels)
    curr_measurement_sum[AUROC] += area_under_roc(output, labels)
    curr_measurement_sum[F1] += f1_score(output, labels)
    curr_measurement_sum[PRECISION] += precision(output, labels)


def feed_model(model, hidden_size, X, y, data_size, batch_size, criterion, optimizer,
               accuracy, area_under_roc, f1_score, precision, measurements, mode, device):
    curr_measurement_sum = {
        LOSS: 0.0,
        ACC: 0.0,
        AUROC: 0.0,
        F1: 0.0,
        PRECISION: 0.0
    }
    amount = 0.0
    # train the model
    for index in range(0, data_size, batch_size):
        amount += 1
        phrases = torch.stack(X[index: index + batch_size]).to(device)
        labels = torch.tensor(np.array(y[index: index + batch_size]), dtype=torch.float).to(device)

        curr_batch_size = len(labels)
        hidden_state = torch.zeros(curr_batch_size, hidden_size).to(device)
        for i in range(curr_batch_size):
            output, hidden_state = model(phrases[:, i, :], hidden_state)

        loss = criterion(output, labels)

        # measure
        _update_curr_measurement_sum(output.view(-1), labels.type(torch.int32).view(-1), loss.item(),
                                     accuracy, area_under_roc, f1_score, precision, curr_measurement_sum)

        # train
        if mode == TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for measurement in measurements.keys():
        measurements[measurement][mode].append(curr_measurement_sum[measurement] / amount)
