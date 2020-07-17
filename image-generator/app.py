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

    experience_file = './data/experience.csv'
    number_of_experiences, ids_for_categories = load_experience_data(experience_file, n_ratings=3)

    number_of_true_images_for_provider = 3
    number_of_noise_images_for_provider = 3
    number_of_images_in_collage = 6
    output_directory = './data/generated-data'
    features_original_images_file = './data/images/features-images-1'

    image_generator = ImageGenerator(images_directory, categories_file, features_original_images_file)
    image_generator.generate(number_of_true_images_for_provider=number_of_true_images_for_provider, number_of_noise_images_for_provider=number_of_noise_images_for_provider, number_of_images=number_of_experiences, ids=ids_for_categories, number_of_images_in_collage=number_of_images_in_collage, output_directory=output_directory)

    evaluator = Evaluator(output_directory, features_file='./data/features-generated-data')
    evaluator.visualize(show='ratings')
    evaluator.classify()


if __name__ == '__main__':
    main()