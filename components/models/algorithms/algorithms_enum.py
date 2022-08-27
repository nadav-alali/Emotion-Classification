from enum import Enum


class ClassifierAlgorithm(Enum):
    LSTM = 1
    GRU = 2
    GRU_SELF_ATTENTION = 3
    LSTM_SELF_ATTENTION = 4
    BIDIRECTIONAL_LSTM = 5
