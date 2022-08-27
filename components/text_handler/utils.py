import re


def clean_sentence(text: str):
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text.split()
