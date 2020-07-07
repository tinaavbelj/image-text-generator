import pandas as pd
import numpy as np


def load_experience_data(experience_file, n_ratings=10):
    data = pd.read_csv(experience_file, sep=';')

    ids = data['Id']
    categories = data['ExperienceGroup']
    ratings = data['Rating']

    unique_categories = list(set(categories))
    unique_ratings = list(set(ratings))
    define_new_ratings = False

    if len(unique_ratings) != n_ratings:
        define_new_ratings = True
        original_n_ratings = len(unique_ratings)
        unique_ratings = [i + 1 for i in range(n_ratings)]

    ids_for_categories = {}
    number_of_experiences = {}
    for category in unique_categories:
        ids_for_categories[category.lower()] = {}
        number_of_experiences[category.lower()] = {}
        for rating in unique_ratings:
            ids_for_categories[category.lower()][rating] = []
            number_of_experiences[category.lower()][rating] = 0

    # Define ratings thresholds (approximately the same number of providers for each new rating)
    if define_new_ratings:
        indexes = [i for i in range(len(ids))]
        sorted_indexes = [x for _, x in sorted(zip(ratings, indexes))]
        sorted_indexes = np.array(sorted_indexes)
        split_indexes = np.array_split(sorted_indexes, n_ratings)
        thresholds = []
        for indexes_partition in split_indexes:
            thresholds.append(ratings[indexes_partition[1]])
        print('\nThresholds: ')
        print(thresholds)
        print()
    thresholds.sort()

    for index, experience_id in enumerate(ids):
        if define_new_ratings:
            #d = original_n_ratings / n_ratings
            #thresholds = [d * (i + 1) for i in range(n_ratings)]
            right_threshold_index = 0
            for t_index, threshold in enumerate(thresholds):
                if ratings[index] >= threshold:
                    right_threshold_index = t_index
            new_rating = right_threshold_index + 1
            ids_for_categories[categories[index].lower()][new_rating].append(experience_id)
            number_of_experiences[categories[index].lower()][new_rating] += 1
        else:
            ids_for_categories[categories[index].lower()][ratings[index]].append(experience_id)
            number_of_experiences[categories[index].lower()][ratings[index]] += 1

    return number_of_experiences, ids_for_categories
