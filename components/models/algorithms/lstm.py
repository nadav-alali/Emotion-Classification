import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, phrase_length=32, bidirectional=False):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        num_layers = 2 if bidirectional else 1
        dropout = 0.3 if bidirectional else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.out = nn.Sequential(nn.Linear(phrase_length * hidden_size * num_layers, hidden_size),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_size, output_size))
        self.softmax = torch.softmax

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.out(out.reshape(out.shape[0], out.shape[1] * out.shape[2]))
        out = self.softmax(out, dim=1)
        return out, hidden

    def get_model_name(self):
        if self.bidirectional:
            return "bidirectional-LSTM"
        return "LSTM"
