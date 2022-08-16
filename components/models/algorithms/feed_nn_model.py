import torch

LOSS = "loss"
TRAIN = "train"
TEST = "test"
RECALL = "recall"
F1_SCORE = "F1 score"
PRECISION = "precision"


def _update_curr_measurement_sum(output, labels, loss, recall, f1, precision,
                                 curr_measurement_sum):
    curr_measurement_sum[LOSS] += loss
    curr_measurement_sum[RECALL] += recall(output, labels)
    curr_measurement_sum[F1_SCORE] += f1(output, labels)
    curr_measurement_sum[PRECISION] += precision(output, labels)


def feed_model(model, dataset, criterion, optimizer, recall, f1, precision, measurements, mode):
    curr_measurement_sum = {
        LOSS: 0.0,
        RECALL: 0.0,
        F1_SCORE: 0.0,
        PRECISION: 0.0
    }
    amount = 0.0
    # train the model
    for phrases, labels in dataset:
        amount += 1
        output, hidden_state = model(phrases)
        loss = criterion(output, labels)

        # measure
        _update_curr_measurement_sum(output.view(-1), labels.type(torch.int32).view(-1), loss.item(),
                                     recall, f1, precision, curr_measurement_sum)

        # backpropagation - only while training
        if mode == TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for measurement in measurements.keys():
        measurements[measurement][mode].append(curr_measurement_sum[measurement] / amount)
