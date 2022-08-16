import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, phrase_length=32):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size * phrase_length, output_size)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        hidden_state = torch.zeros(1, x.shape[0], self.hidden_size)
        cell_state = torch.zeros(1, x.shape[0], self.hidden_size)
        out, hidden = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out.reshape(out.shape[0], out.shape[1] * out.shape[2]))
        out = self.sigmoid(out)
        return out, hidden

    def get_model_name(self):
        return "LSTM"
