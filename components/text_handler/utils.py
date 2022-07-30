import nltk
from nltk.corpus import stopwords
import string


def clean_sentence(text: str, special_stop_words: list):
    text_without_special_stop_words = text[:]
    # remove special stop words
    for special_stop_word in special_stop_words:
        text_without_special_stop_words = text_without_special_stop_words.replace(special_stop_word, "")
    # split into words
    tokens = nltk.word_tokenize(text_without_special_stop_words)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words and also use Porter algorithm to get the stem of a word
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words
