from components.models.algorithms.algorithms_enum import ClassifierAlgorithm
from components.models.algorithms.gru import GRU
from components.models.algorithms.lstm import LSTM
import torch.nn as nn
from components.models.algorithms.self_attention import SelfAttention


def algorithms_factory(algorithm: ClassifierAlgorithm, input_size, output_size, hidden_size, device):
    if algorithm == ClassifierAlgorithm.GRU:
        return GRU(input_size, output_size, hidden_size).to(device)
    elif algorithm == ClassifierAlgorithm.LSTM:
        return LSTM(input_size, output_size, hidden_size).to(device)
    elif algorithm == ClassifierAlgorithm.GRU_SELF_ATTENTION:
        return SelfAttention(nn.GRU, input_size, hidden_size, output_size, "GRU")
    elif algorithm == ClassifierAlgorithm.LSTM_SELF_ATTENTION:
        return SelfAttention(nn.LSTM, input_size, hidden_size, output_size, "LSTM")
    elif algorithm == ClassifierAlgorithm.BIDIRECTIONAL_LSTM:
        return LSTM(input_size, output_size, hidden_size, bidirectional=True).to(device)
