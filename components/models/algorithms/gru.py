import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden_update = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden_reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.activation = torch.tanh
        self.sigmoid = nn.functional.sigmoid
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state_):
        concat_inputs = torch.cat((x, hidden_state_), 1)
        update_gate = self.sigmoid(self.in2hidden_update(concat_inputs))
        reset_gate = self.sigmoid(self.in2hidden_reset(concat_inputs))

        concat_hidden_reset_x = torch.cat((x, reset_gate * hidden_state_), 1)
        hidden_tilda = self.hidden_layer(concat_hidden_reset_x)
        hidden_tilda = self.activation(hidden_tilda)

        hidden = torch.multiply(1 - update_gate, hidden_state_) + torch.multiply(update_gate, hidden_tilda)
        out = self.sigmoid(self.out(hidden))
        return out, hidden
