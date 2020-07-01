from nltk.corpus import wordnet
import json
import numpy as np
import os


class WordsLoader:
    def __init__(self, output_directory, categories_file):
        self.output_directory = output_directory
        self.categories_file = categories_file

    def load(self):
        """
        Generates words using wordnet and words specified in categories_file and saves them in self.output_directory

        :param current_id: Id for objects for test set
        :returns:  train X, test X, train y,  test y
        """
        with open(self.categories_file, 'r') as file:
            categories_data = json.load(file)

        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)

        for category in categories_data:
            # Add words defined in categories.json
            words = category['text_additional_words'] if category['text_additional_words'] else []

            # Add words from WordNet for all keywords
            for query in category['text_keywords']:

                word = wordnet.synsets(query)

                for synset in word:
                    for lemma in synset.lemmas():
                        new_word = lemma.name().replace('_', ' ')
                        words.append(new_word)

                for ss in word:
                    for hyper in ss.hypernyms():
                        new_word = hyper.name().split('.')[0].replace('_', ' ')
                        words.append(new_word)

            # Keep only unique words
            words = list(set(words))

            # Save words to file
            np.savetxt(f'{self.output_directory}/{category["name"]}.csv', words, fmt="%s", delimiter="\n")
