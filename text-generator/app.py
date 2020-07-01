import os

from words_loader import WordsLoader
from text_generator import TextGenerator
from evaluator import Evaluator


def main():
    # Parameters
    words_directory = './data/words/'
    categories_file = '../categories.json'
    neutral_words_file = './data/neutral_words.csv'

    if not os.path.isdir(words_directory):
        loader = WordsLoader(output_directory=words_directory, categories_file=categories_file)
        loader.load()

    number_of_texts = {
        'leisure': {
            1: 2,
            2: 3,
            3: 2
        },
        'adventure': {
            1: 5,
            2: 2,
            3: 1
        }
    }

    ids = {
        'leisure': {
            1: [1, 2],
            2: [3, 5, 6],
            3: [7, 8]
        },
        'adventure': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        }
    }

    text_length = 100
    noise_ratio = 0.3
    neutral_words_ratio = 0.2
    output_directory = './generated-data2'

    #text_generator = TextGenerator(words_directory, neutral_words_file, categories_file)
    #text_generator.generate(number_of_texts, ids, text_length, noise_ratio, neutral_words_ratio, output_directory)

    evaluator = Evaluator(output_directory)
    evaluator.visualize(show='ratings')
    evaluator.classify(algorithm='knn')


if __name__ == '__main__':
    main()