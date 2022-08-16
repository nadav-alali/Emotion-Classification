from components.datasets.dataset_enum import Dataset
from components.datasets.get_go_emotions_dataset import GoEmotionsDataset
from components.text_handler.embedding.embedding import Embedding


def dataset_factory(dataset_type: Dataset, embedding: Embedding, sampler=lambda x: x):
    if dataset_type == Dataset.GO_EMOTIONS:
        return GoEmotionsDataset(embedding, sampler)
