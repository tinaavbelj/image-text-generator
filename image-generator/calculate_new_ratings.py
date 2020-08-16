import numpy as np
import pandas as pd


def load_ratings_data(booking_file, n_experiences, ids_indexes):
    """
    Loads data for ratings to matrix

    :param booking_file: path to csv file with ratings
    :param n_experiences: number of all experiences
    :param ids_indexes: keys - experience ids, values - indexes for rows in matrix
    :returns: ratings matrix
    """
    data_booking = pd.read_csv(booking_file, sep=';')
    experience_ids_booking = list(data_booking['UserId'])
    user_ids = list(data_booking['Rating'])
    experience_ratings = list(data_booking['TimeOfDay'])

    # Ratings matrix
    unique_user_ids = list(set(user_ids))
    n_users = len(unique_user_ids)
    ratings_matrix = np.zeros(shape=(n_users, n_experiences))

    for current_user_id in unique_user_ids:
        # Find indexes for this user
        indexes = []
        for i, user_id in enumerate(user_ids):
            if user_id == current_user_id:
                indexes.append(i)

        # Add data for current user to ratings matrix
        new_user_id = current_user_id - 1

        for index in indexes:
            # If this experience exists in generated pictures
            if int(experience_ids_booking[index]) in ids_indexes.keys():
                new_experience_id = ids_indexes[experience_ids_booking[index]]
                ratings_matrix[new_user_id][new_experience_id] = experience_ratings[index]

    return ratings_matrix


def calculate_experience_ratings(ratings_matrix):
    """
    Calculates experience ratings from average user ratings

    :param ratings_matrix: matrix of user ratings
    :returns:  vector of new ratings
    """

    n_users = ratings_matrix.shape[0]
    n_experiences = ratings_matrix.shape[1]

    # Calculate average for each user (only non-zero values)
    users_avg = []
    for i in range(n_users):
        ratings_sum = 0
        n = 0
        for j in range(n_experiences):
            if ratings_matrix[i, j] != 0:
                ratings_sum += ratings_matrix[i, j]
                n += 1
        average = ratings_sum / n
        users_avg.append(average)

    # Subtract user average from each rating
    average_ratings_matrix = np.zeros(ratings_matrix.shape)
    for i in range(n_users):
        for j in range(n_experiences):
            if ratings_matrix[i, j] != 0:
                new_rating = ratings_matrix[i, j] - users_avg[i]
                average_ratings_matrix[i, j] = new_rating

    # Calculate average for each experience
    experience_ratings = []
    for j in range(n_experiences):
        ratings_sum = 0
        n = 0
        for i in range(n_users):
            if average_ratings_matrix[i, j] != 0:
                ratings_sum += average_ratings_matrix[i, j]
            if ratings_matrix[i, j] != 0:
                n += 1
        average = ratings_sum / n
        if average > 0:
            experience_ratings.append(2)
        else:
            experience_ratings.append(1)

    return experience_ratings


def load_data(booking_file, experience_file, results_file):
    data = pd.read_csv(experience_file, sep=';')
    ids = data['Id']
    categories = data['ExperienceGroup']
    ratings = data['Rating']

    # Ids to column indexes
    new_column_index = 0
    ids_indexes = {}
    for i in sorted(list(set(ids))):
        ids_indexes[int(i)] = new_column_index
        new_column_index += 1

    n_experiences = len(ids)

    ratings_matrix = load_ratings_data(booking_file, n_experiences, ids_indexes)

    experience_ratings = calculate_experience_ratings(ratings_matrix)

    # Save new experience file
    data = {'Id': ids,
            'ExperienceGroup': categories,
            'Rating': experience_ratings}
    df = pd.DataFrame(data, columns=['Id', 'ExperienceGroup', 'Rating'])
    df.to_csv(results_file, sep=';')


def main():
    # Parameters

    experience_file = './data/experience.csv'
    booking_file = './data/booking.csv'
    results_file = './data/experience-new-ratings.csv'

    load_data(booking_file, experience_file, results_file)


if __name__ == '__main__':
    main()