from components.models.algorithms.algorithms_enum import ClassifierAlgorithm
from components.models.algorithms.gru import GRU
from components.models.algorithms.lstm import LSTM


def algorithms_factory(algorithm: ClassifierAlgorithm, input_size, output_size, hidden_size, device):
    if algorithm == ClassifierAlgorithm.GRU:
        return GRU(input_size, output_size, hidden_size).to(device)
    elif algorithm == ClassifierAlgorithm.LSTM:
        return LSTM(input_size, output_size, hidden_size).to(device)