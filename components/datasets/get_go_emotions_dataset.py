from components.datasets.dataset_enum import Dataset
from components.datasets.get_data import GetData
import wget
import os
import pandas as pd

from components.datasets.utils import get_data_folder_path
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

STOP_WORDS = ["[NAME]"]  # when a name appears in a reddit post Google replaced it with this string


def _get_file_name(file_number):
    return f"{OUTPUT_FILE_NAME}_{file_number}.{OUTPUT_FILE_TYPE}"


class GoEmotionsDataset(GetData):

    def __init__(self, embedding: Embedding):
        super().__init__(Dataset.GO_EMOTIONS, embedding)
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
            phrases = [clean_sentence(sentence, STOP_WORDS) for sentence in list(data["text"])]
            embedded_phrases = [self.embedding.embed(phrase) for phrase in phrases]
            labels = list(data.loc[:, LABELS[0]:].to_numpy())
            self.data.extend(embedded_phrases)
            self.labels.extend(labels)
        self.save_data()

    def get_text_label_from_label_vector(self, label_vector: list) -> str:
        string_label = ''
        for i, label in enumerate(label_vector):
            if label:
                string_label += f'{LABELS[i]}, '
        return string_label[: -2]

# example:
# from components.text_handler.embedding.glove_embedding import GloveEmbedding
# glove_embedding = GloveEmbedding()
# ds = GoEmotionsDataset(glove_embedding)
