import numpy as np
from components.datasets.dataset_enum import Dataset
from components.datasets.get_data import GetData
import wget
import os
import pandas as pd

from components.datasets.utils import get_data_folder_path
from components.models.utils import balance_data
from components.text_handler.embedding.embedding import Embedding
from components.text_handler.utils import clean_sentence

FIRST_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv"
SECOND_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv"
THIRD_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"

URLS = [FIRST_URL, SECOND_URL, THIRD_URL]

OUTPUT_FILE_NAME = "go_emotions"
OUTPUT_FILE_TYPE = "csv"

LABELS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
          'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
          'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
          'relief', 'remorse', 'sadness', 'surprise', 'neutral']

LABELS_TO_REMOVE = ["grief", "nervousness", "relief", "pride"]

NUM_OF_NEUTRAL_LABELS = 22520

def _get_file_name(file_number):
    return f"{OUTPUT_FILE_NAME}_{file_number}.{OUTPUT_FILE_TYPE}"


class GoEmotionsDataset(GetData):

    def __init__(self, embedding: Embedding, sampler):
        super().__init__(Dataset.GO_EMOTIONS, embedding, sampler)
        self.load_data()

    def is_data_exists(self) -> bool:
        folder_path = get_data_folder_path()
        for i in range(len(URLS)):
            path = os.path.join(folder_path, _get_file_name(i))
            if not os.path.exists(path):
                return False
        return True

    def load_data(self):
        if len(self.data) > 0:
            return

        folder_path = get_data_folder_path()
        output_paths = [os.path.join(folder_path, _get_file_name(i)) for i in range(len(URLS))]
        if not self.is_data_exists():
            for i, url in enumerate(URLS):
                wget.download(url, output_paths[i])

        for output_path in output_paths:
            data = pd.read_csv(output_path)
            data = data[data["example_very_unclear"] == False]  # only take labeled data
            # remove filtered labels
            for l in LABELS_TO_REMOVE:
                data = data[data[l] == 0]
            data = data.drop(columns=LABELS_TO_REMOVE)
            phrases = [clean_sentence(sentence) for sentence in list(data["text"])]
            labels = list(data.loc[:, LABELS[0]:].to_numpy())
            valid_indexes = [i for i, _ in enumerate(phrases) if len(phrases[i]) > 0 and np.sum(labels[i]) == 1]
            phrases = [phrases[i] for i in valid_indexes]
            embedded_phrases = [self.embedding.embed(phrase) for phrase in phrases]
            labels = [labels[i] for i in valid_indexes]
            self.phrases.extend(phrases)
            self.data.extend(embedded_phrases)
            self.labels.extend(labels)
        # under-sample neutral label
        neutral_labels_indices = [i for i, l in enumerate(self.labels) if l[-1] == 1]
        sampled_neutral_indices = np.random.choice(neutral_labels_indices, NUM_OF_NEUTRAL_LABELS)
        labels_to_filter = set(neutral_labels_indices) - set(list(sampled_neutral_indices))
        under_sampled_phrases, under_sampled_data, under_sampled_labels = [], [], []
        for i in range(len(self.data)):
            if i not in labels_to_filter:
                under_sampled_phrases.append(self.phrases[i])
                under_sampled_data.append(self.data[i])
                under_sampled_labels.append(self.labels[i])
        self.phrases, self.data, self.labels = balance_data(under_sampled_phrases, under_sampled_data, under_sampled_labels)
        self.save_data()

    def get_text_label_from_label_vector(self, label_vector: list) -> str:
        string_label = ''
        for i, label in enumerate(label_vector):
            if label:
                string_label += f'{LABELS[i]}, '
        return string_label[: -2]
