import torch
from torchtext.vocab import GloVe

from components.text_handler.embedding.embedding import Embedding

EMBEDDING_SIZE = 100
MAX_LENGTH = 32


class GloveEmbedding(Embedding):
    def __init__(self):
        self.model = GloVe(name='6B', dim=EMBEDDING_SIZE)

    def embed(self, sentence):
        embedded = self.model.get_vecs_by_tokens(sentence)
        if embedded.shape[0] != EMBEDDING_SIZE or embedded.shape[1] != EMBEDDING_SIZE:
            embedded = torch.nn.functional.pad(embedded, (0, 0, 0, MAX_LENGTH - embedded.shape[0]))
        return torch.unsqueeze(embedded, 0)
