import os
from collage_maker import make_collage
import random
import json
from utils import load_image
from utils import get_features
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from shutil import copyfile


def flatten_list(x):
    return sum(x, [])


class ImageGenerator:
    def __init__(self, data_directory, categories_file, features_original_images_file):
        self.data_directory = data_directory
        self.categories_file = categories_file
        self.data = {}
        self.features_original_images_file = features_original_images_file
        self.features_original_images = []
        self.paths_original_images = []
        self.number_of_images = {}
        self.ratings = []
        self.number_of_images_in_collage = 0
        self.output_directory = ''
        self.tmp_output_directory = ''
        self.number_of_true_images_for_provider = 0
        self.number_of_noise_images_for_provider = 0
        self.ids = {}

        #file_to_delete = data_directory + '/.DS_Store'
        #os.remove(file_to_delete)

        with open(self.categories_file, 'r') as file:
            categories_data = json.load(file)
        categories = []

        for category in categories_data:
            categories.append(category['name'])

        self.categories = categories

        for category in categories:
            images_directory = data_directory + '/' + category
            self.data[category] = [os.path.join(images_directory, image_path)
                                   for image_path in os.listdir(images_directory) if not image_path.endswith(".DS_Store")]

    def get_other_categories(self, category):
        categories = self.categories.copy()
        categories.remove(category)
        return categories

    def select_best_original_images(self):
        """
        Select original images with the best silhouette scores (label is category). Save selected features to
        self.features_original_images, paths to selected images to self.paths_original_images, categories to
        self.categories_original_images
        """
        print('Select the best original images\n')
        data_all = []
        labels = []

        # Delete invalid format images
        for category in self.categories:
            category_data = os.listdir(self.data_directory + '/' + category)
            for image in category_data:
                try:
                    load_image(self.data_directory + '/' + category + '/' + image)
                except:
                    # Delete this image
                    print('Removing file')
                    os.remove(self.data_directory + '/' + category + '/' + image)
            data_all = data_all + [self.data_directory + '/' + category + '/' + image for image in
                                   os.listdir(self.data_directory + '/' + category)]
            labels = labels + [category for _ in os.listdir(self.data_directory + '/' + category)]

        features = get_features(self.features_original_images_file, data_all)

        print('\nCalculate silhouette scores\n')
        scores = silhouette_samples(features, labels)

        # Remove images below silhouette score threshold for each category
        self.features_original_images = {}
        self.paths_original_images = {}
        for category in self.categories:
            category_indexes = []
            category_labels = []
            category_scores = []
            category_paths = []
            for index, label in enumerate(labels):
                if label == category:
                    category_indexes.append(index)
                    category_labels.append(label)
                    category_scores.append(scores[index])
                    category_paths.append(data_all[index])

            sorted_scores_indexes = [r for _, r in
                                     sorted(zip(category_scores, [i for i in range(len(category_scores))]))]

            # Keep best 50 images for each category
            best_indexes = sorted_scores_indexes[-50:]
            self.features_original_images[category] = features[best_indexes, :]
            self.paths_original_images[category] = [path for index, path in enumerate(category_paths) if index in best_indexes]

    def get_number_of_other_providers(self, category):
        other_categories = self.get_other_categories(category)
        n = 0
        for c in other_categories:
            ratings = list(self.number_of_images[c].keys())
            for r in ratings:
                n += self.number_of_images[category][r]
        return n

    def generate_collage_images(self):
        """
        Generate collage images and save them to output_directory
        """
        print('Generate collage images\n')
        # self.output_directory + '-tmp' is a folder with collages that will be used later for selection of images for the final dataset
        output_directory = self.output_directory + '-tmp'
        self.tmp_output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        else:
            return

        for category in self.categories:
            # select only current category features
            current_features = self.features_original_images[category]
            selected_img_names = self.paths_original_images[category]

            n_ratings = len(self.ratings)
            kmeans = KMeans(n_clusters=n_ratings, random_state=0).fit(current_features)
            labels = kmeans.labels_
            # Labels should match ratings
            labels = [int(label + 1) for label in labels]

            images_indexes = {}

            for l in (list(set(labels))):
                images_indexes[l] = []

            for index, label in enumerate(labels):
                images_indexes[label].append(index)

            other_n = self.get_number_of_other_providers(category)
            for rating in self.ratings:
                # Generate more images to later keep only the ones with the best silhouette score
                for i in range(self.number_of_images[category][rating] * (self.number_of_true_images_for_provider + other_n)): # +len(self.ratings)
                    output_path = f'{output_directory}/{category}-{rating}-{i}.png'
                    n_images = self.number_of_images_in_collage if len(images_indexes[rating]) >= self.number_of_images_in_collage else len(images_indexes[rating])
                    selected_images_indexes = random.sample(images_indexes[rating], k=n_images)
                    images = [selected_img_names[current_index] for current_index in selected_images_indexes]
                    make_collage(images, output_path, width=450, init_height=150)

    def generate_final_dataset(self):
        """
        Generate final dataset of collage images
        """
        print('Generate final dataset\n')
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        else:
            print('\nOutput directory already exists\n')
            return
        features_file = './data/features-tmp'
        file_names = os.listdir(self.tmp_output_directory)
        images_paths = [self.tmp_output_directory + '/' + name for name in file_names]
        images_categories = [name.split('-')[0] for name in file_names]

        features = get_features(features_file, images_paths)

        # Find cluster of images for each rating and keep best images for each category and rating
        best_images = {}
        for current_category in self.categories:
            # Select only current category features
            indexes = []
            selected_images_paths = []
            for index, category in enumerate(images_categories):
                if category == current_category:
                    indexes.append(index)
                    selected_images_paths.append(images_paths[index])
            category_features = features[indexes, :]

            current_ratings = list(self.number_of_images[current_category].keys())
            n_ratings = len(current_ratings)
            kmeans = KMeans(n_clusters=n_ratings, random_state=0).fit(category_features)
            labels = kmeans.labels_
            new_labels = [i + 1 for i in labels]
            labels = new_labels
            scores = silhouette_samples(category_features, labels)

            # Keep best images for each category and rating
            best_images[current_category] = {}
            other_n = self.get_number_of_other_providers(category)
            for r in list(self.number_of_images[category].keys()):
                scores_for_rating = []
                indexes_for_rating = []
                for index, label in enumerate(labels):
                    if label == r:
                        scores_for_rating.append(scores[index])
                        indexes_for_rating.append(index)
                sorted_scores_indexes = [r for _, r in sorted(zip(scores_for_rating, [i for i in range(len(scores_for_rating))]))]
                n = self.number_of_images[current_category][r] * self.number_of_true_images_for_provider
                best_indexes = sorted_scores_indexes[-n:]
                best_images[current_category][r] = [path for index, path in enumerate(selected_images_paths) if index in best_indexes]

        # Select images for each provider
        new_provider_id = 1
        for current_category in self.categories:
            other_categories = self.get_other_categories(category)
            for r in list(self.number_of_images[current_category].keys()):

                # Create providers for current category and rating
                next_id_index = 0
                for i in range(self.number_of_images[current_category][r]):
                    # Select true images
                    true_selected_images = random.sample(best_images[current_category][r], k=self.number_of_true_images_for_provider)

                    # Select noise images
                    other_category = random.choice(other_categories)
                    noise_selected_images = random.sample(best_images[other_category][r], k=self.number_of_noise_images_for_provider)

                    all_selected_images = true_selected_images + noise_selected_images

                    # Specify new provider's id
                    if self.ids:
                        provider_id = self.ids[current_category][r][next_id_index]
                        next_id_index += 1
                    else:
                        provider_id = new_provider_id
                        new_provider_id += 1

                    # Copy images
                    for image_index, image in enumerate(all_selected_images):
                        new_file = f"{self.output_directory}/{provider_id}-{current_category}-{r}-{image_index}.png"
                        old_file = image
                        copyfile(old_file, new_file)

    def generate(self, number_of_true_images_for_provider, number_of_noise_images_for_provider, number_of_images, ids, number_of_images_in_collage, output_directory):
        """
        Generate data set of collage images and save it to output_directory
        """
        # Save values
        self.number_of_images_in_collage = number_of_images_in_collage
        self.number_of_images = number_of_images
        self.categories = list(number_of_images.keys())
        self.ratings = list(number_of_images[self.categories[0]].keys())
        self.output_directory = output_directory
        self.number_of_true_images_for_provider = number_of_true_images_for_provider
        self.number_of_noise_images_for_provider = number_of_noise_images_for_provider
        self.ids = ids

        # Generate
        self.select_best_original_images()
        self.generate_collage_images()
        self.generate_final_dataset()
