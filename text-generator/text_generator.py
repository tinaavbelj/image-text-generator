import pandas as pd
import numpy as np
import random
import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def flatten_list(x):
    return sum(x, [])


class TextGenerator:
    def __init__(self, data_directory, neutral_words_file, categories_file):
        self.data_directory = data_directory
        self.neutral_words_file = neutral_words_file

        categories = []
        with open(categories_file, 'r') as file:
            categories_data = json.load(file)
        for category in categories_data:
            categories.append(category['name'])

        self.categories = categories

        self.data = {}
        self.neutral_words = None

        self.read_data()

    def read_data(self):
        # Categories words to array
        for category in self.categories:
            df = pd.read_csv(self.data_directory + category + '.csv', header=None)
            self.data[category] = df.iloc[:, 0].values

        # Neutral words to array
        df = pd.read_csv(self.neutral_words_file, header=None)
        self.neutral_words = df.iloc[:, 0].values

    def get_other_categories(self, category):
        categories = self.categories.copy()
        categories.remove(category)
        return categories

    def write_text_to_file(self, output_directory, filename, text):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(f"{output_directory}/{filename}", "w") as file:
            file.write(text)

   # def divide_words_for_this_category(self, category, rating, all_ratings):
        #current_rating_words = round(len(self.data[category]) / len(number_of_texts[category][rating]))
        #other_ratings_words = len(self.data[category]) - current_rating_words

        #n_other_ratings = len(all_ratings) - 1
        #n_words_for_each_rating = len(self.data[category]) / len(all_ratings)

        #unit_probability = 100 / (2 * n_words_for_each_rating + n_other_ratings * n_words_for_each_rating)
        #all_ratings = len(number_of_texts[category].keys())

        #start_index = (rating - 1) * n_words_for_each_rating
        #end_index = start_index + n_words_for_each_rating
        #words_current_rating = self.data[category][start_index:end_index]
        #words_other_rating = self.data[category][0:start_index]

    def generate(self, number_of_texts, ids, text_length, noise_ratio, neutral_words_ratio, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        new_id = 0
        for category in number_of_texts.keys():
            all_ratings = number_of_texts[category].keys()
            words_for_ratings = np.array_split(np.array(self.data[category]), len(all_ratings))

            other_categories = self.get_other_categories(category)

            current_ratio = 1 - noise_ratio - neutral_words_ratio
            other_ratio = noise_ratio / len(other_categories)

            for rating_index, rating in enumerate(number_of_texts[category].keys()):
                if not ids:
                    pass

                words_for_other_ratings = flatten_list([words.tolist() for index, words in enumerate(words_for_ratings) if index != rating_index])

                for text_number in range(number_of_texts[category][rating]):

                    words_current_category = flatten_list([
                        # current
                        random.choices(words_for_ratings[rating_index], k=round(text_length * current_ratio / 2)),
                        # current category, wrong rating
                        random.choices(words_for_other_ratings[rating_index], k=round(text_length * current_ratio / 2)),
                        # neutral
                        random.choices(self.neutral_words, k=round(text_length * neutral_words_ratio))
                    ])

                    random.shuffle(words_current_category)

                    other_category = random.choice(other_categories)

                    words_other_category = flatten_list([
                        # neutral
                        random.choices(self.neutral_words, k=round(text_length * other_ratio * neutral_words_ratio)),
                        # other
                        random.choices(self.data[other_category], k=round(text_length * other_ratio)),
                    ])

                    random.shuffle(words_other_category)

                    words = flatten_list([
                        # neutral
                        words_current_category,
                        # other
                        words_other_category
                    ])

                    text = ' '.join(words)

                    if ids:
                        i = ids[category][rating][text_number]
                    else:
                        i = new_id
                        new_id += 1

                    self.write_text_to_file(output_directory, f"/{i}-{category}-{rating}.txt", text)

            # Find clusters for each rating
            #kmeans = KMeans(n_clusters=3, random_state=0).fit(features_category)
            #labels = kmeans.labels_
            #new_labels = [i + 1 for i in labels]
            #labels = new_labels
            #scores = silhouette_samples(features_category, labels)
