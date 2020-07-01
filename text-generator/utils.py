import numpy as np
from gensim.models import Word2Vec


def text_to_list(file):
    """
    Read text from the file to a list

    :param file: path to text
    :returns:  list of words in the text
    """
    with open(file, 'r') as f:
        text = f.read()
    words = text.split()
    return words


def text_to_vector(text, model):
    """
    Calculate vector for text as average of vectors of all words in the text

    :param text: List of words in a text
    :param model: Word2Vec model made on all texts in the dataset
    :returns:  vector representing text
    """
    vector = np.zeros(100)
    num_words = 0
    for word in text:
        try:
            vector = np.add(vector, model[word])
            num_words += 1
        except:
            pass
    return vector / np.sqrt(vector.dot(vector))


def texts_to_vectors(texts_paths):
    """
    Calculate matix of features for texts_paths (array of paths)

    :param texts_paths: List of paths to texts
    :returns:  features
    """

    all_texts = []
    for name in texts_paths:
        current_text = text_to_list(name)
        all_texts.append(current_text)

    model = Word2Vec(all_texts, size=100, min_count=1)

    features = []
    for text in all_texts:
        features.append(text_to_vector(text, model))

    features = np.asarray(features)

    return features

