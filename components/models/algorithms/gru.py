import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, phrase_length=32):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.phrase_length = phrase_length
        self.gru = nn.GRU(input_size, hidden_size, 1)
        self.sigmoid = torch.sigmoid
        self.softmax = torch.softmax
        self.out = nn.Linear(phrase_length * hidden_size, output_size)

    def forward(self, x):
        h0 = torch.torch.zeros(1, x.shape[1], self.hidden_size)
        out, hidden = self.gru(x, h0)
        out = self.out(out.reshape(out.shape[0], self.phrase_length * self.hidden_size))
        # out = self.sigmoid(out)
        out = self.softmax(out, dim=1)
        return out, hidden

    def get_model_name(self):
        return "GRU"
