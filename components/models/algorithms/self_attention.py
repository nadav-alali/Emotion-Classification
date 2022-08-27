import torch
import torch.nn as nn

ATTENTION_STRING = "self-attention based"


class SelfAttention(nn.Module):
    def __init__(self, rnn_model, input_size, hidden_size, output_size, rnn_type):
        super(SelfAttention, self).__init__()
        self.encoder = rnn_model(input_size, hidden_size, 1, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size * 32, hidden_size),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_size, output_size))
        self.softmax = torch.softmax
        self.rnn_type = rnn_type

    def forward(self, input):
        outputs, _ = self.encoder(input)
        attn_output, attn_output_weights = self.attention(outputs, outputs, outputs)
        att_x, att_y, att_z = attn_output.shape
        output = self.decoder(attn_output.reshape(att_x, att_y * att_z))
        output = self.softmax(output, dim=1)
        return output, attn_output_weights

    def get_model_name(self):
        return f'{ATTENTION_STRING} {self.rnn_type}'
