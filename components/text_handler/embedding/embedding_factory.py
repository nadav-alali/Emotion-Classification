from components.text_handler.embedding.embedding_enum import EmbeddingType
from components.text_handler.embedding.glove_embedding import GloveEmbedding


def embedding_factory(embedding_type: EmbeddingType):
    if embedding_type == EmbeddingType.GLOVE:
        return GloveEmbedding()
