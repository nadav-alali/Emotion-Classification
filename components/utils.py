import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, phrases, embedded_phrases, labels):
        super(MyDataset, self).__init__()
        self.phrases = phrases
        self.embedded_phrases = torch.from_numpy(np.array([x for x in embedded_phrases]))
        self.labels = torch.from_numpy(np.array(labels)).long()

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        return self.phrases[idx], self.embedded_phrases[idx], self.labels[idx]


def collect_batch(batch):
    phrases_list, embedded_phrases_list, label_list = [], [], []
    for phrase, embedded_phrase, label in batch:
        phrases_list.append(phrase)
        embedded_phrases_list.append(embedded_phrase)
        label_list.append(label)
    embedded_phrases = torch.cat(embedded_phrases_list).reshape(len(phrases_list), 32, 100)
    return phrases_list, embedded_phrases, torch.stack(label_list).float()
