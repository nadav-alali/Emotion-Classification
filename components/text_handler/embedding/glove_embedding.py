from torchtext.vocab import GloVe

from components.text_handler.embedding.embedding import Embedding

EMBEDDING_SIZE = 100


class GloveEmbedding(Embedding):
    def __init__(self):
        self.model = GloVe(name='6B', dim=EMBEDDING_SIZE)

    def embed(self, sentence) -> list:
        return [self.model.get_vecs_by_tokens(word) for word in sentence]
