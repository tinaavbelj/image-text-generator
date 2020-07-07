import os

from image_loader import ImageLoader
from image_generator import ImageGenerator
from evaluator import Evaluator
from load_data import load_experience_data


def main():
    # Parameters
    images_directory = './data/images'
    categories_file = '../categories.json'

    if not os.path.isdir(images_directory):
        loader = ImageLoader(output_directory=images_directory, categories_file=categories_file)
        loader.load()

    number_of_images = {
        'leisure': {
            1: 2,
            2: 3,
            3: 2
        },
        'adventure': {
            1: 5,
            2: 2,
            3: 1
        },
        'business': {
            1: 5,
            2: 2,
            3: 1
        },
        'health or medical': {
            1: 5,
            2: 2,
            3: 1
        },
        'sport and recreation': {
            1: 5,
            2: 2,
            3: 1
        },
        'religious': {
            1: 5,
            2: 2,
            3: 1
        },
        'cultural': {
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
        },
        'business': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        },
        'health or medical': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        },
        'religious': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        },
        'cultural': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        },
        'sport and recreation': {
            1: [9, 10, 11, 12, 4],
            2: [13, 15],
            3: [100]
        }
    }

    experience_file = './data/experience.csv'
    number_of_experiences, ids_for_categories = load_experience_data(experience_file, n_ratings=2)

    number_of_true_images_for_provider = 6
    number_of_noise_images_for_provider = 0
    number_of_images_in_collage = 6
    output_directory = './data/generated-data-2-ratings-0-noise'
    features_original_images_file = './data/images/features-images-1'

    #image_generator = ImageGenerator(images_directory, categories_file, features_original_images_file)
    #image_generator.generate(number_of_true_images_for_provider=number_of_true_images_for_provider, number_of_noise_images_for_provider=number_of_noise_images_for_provider, number_of_images=number_of_experiences, ids=ids_for_categories, number_of_images_in_collage=number_of_images_in_collage, output_directory=output_directory)

    evaluator = Evaluator(output_directory, features_file='./data/features-generated-data-2-ratings-0-noise')
    #evaluator.visualize(show='ratings')
    evaluator.classify()


if __name__ == '__main__':
    main()