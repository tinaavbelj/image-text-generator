import os
from helpers import timestamp
import google_api
import json


class ImageLoader:
    def __init__(self, output_directory, categories_file):
        self.output_directory = output_directory
        self.categories_file = categories_file

    def load(self, n_images_for_keyword=15):
        """
        Loads images using google api for search words specified in categories_file and saves them in self.output_directory

        :param n_images_for_keyword: Number of images loaded for every keyword from categories_file
        """
        with open(self.categories_file, 'r') as file:
            categories_data = json.load(file)

        for category in categories_data:
            images_dir = self.output_directory + '/' + category['name']
            n_images = n_images_for_keyword

            # Make directory if it doesn't exist
            if not os.path.exists(images_dir):
                print(timestamp() + ' Creating the directory "' + images_dir + '" and downloading the content')
                os.makedirs(images_dir)

            # Load images for all queries
            for query in category["images_keywords"]:
                google_api.run(query, images_dir, n_images=n_images)
