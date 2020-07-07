import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import get_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.metrics import accuracy_score


COLORS_RATINGS = {1: 'yellow', 2: 'orange', 3: 'red'}
COLORS_CATEGORIES = {'leisure': 'yellow', 'adventure': 'red', 'religious': 'green', 'business': 'blue', 'sport and recreation': 'pink', 'health or medical': 'purple', 'cultural': 'coral'}


class Evaluator:
    def __init__(self, data_directory, features_file=''):
        self.data_directory = data_directory
        self.features_file = features_file

    def visualize(self, show='ratings', visualization_algorithm='tsne'):
        """
        Draw features in 2D space using pca or tsne, color for categories or ratings

        :param show: show categories or ratings with different colors on the graph
        :param visualization_algorithm: pca or tsne
        """
        print('\nVisualize\n')
        file_names = os.listdir(self.data_directory)
        name_vector = [self.data_directory + '/' + name for name in file_names]
        categories = [name.split("-")[1] for name in file_names]
        ratings = [int(name.split(".")[0].split("-")[2]) for name in file_names]
        features = get_features(self.features_file, name_vector)

        if show == 'categories':
            colors = COLORS_CATEGORIES
            title = 'Kategorije'
        else:
            colors = COLORS_RATINGS
            title = 'Ocene'

        if visualization_algorithm == 'pca':
            pca = PCA(n_components=2)
            components = pca.fit_transform(features)
        else:
            tsne = TSNE(n_components=2)
            components = tsne.fit_transform(features)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=12)

        for index, point in enumerate(components):
            if show == 'categories':
                color = colors[categories[index]]
            else:
                color = colors[ratings[index]]
            ax.plot(point[0], point[1], marker='o', markersize=3, color=color)
        ax.grid()
        custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
        ax.legend(custom_lines, colors.keys())
        plt.show()

    def classify(self):
        """
        Classification of images with K nearest neighbors. Print classification accuracy and plot confusion matrix.
        """
        print('\nClassify\n')
        file_names = os.listdir(self.data_directory)
        name_vector = [self.data_directory + '/' + name for name in file_names]
        ids_vector = [name.split('-')[0] for name in file_names]
        ratings_vector = [int(name.split('-')[-2]) for name in file_names]
        features = get_features(self.features_file, name_vector)

        unique_ratings = list(set(ratings_vector))
        unique_ids = list(set(ids_vector))

        true_ratings = {}
        predicted_ratings = {}
        predicted_ratings_vector = []
        true_ratings_vector = []

        for current_id in unique_ids:
            # Images for current_id to test set and other images to train set
            test_indexes = []
            train_indexes = []
            for index, img_id in enumerate(ids_vector):
                if img_id == current_id:
                    test_indexes.append(index)
                else:
                    train_indexes.append(index)

            train_X = features[train_indexes, :]
            test_X = features[test_indexes, :]

            train_y = [ratings_vector[j] for j in train_indexes]
            test_y = [ratings_vector[j] for j in test_indexes]

            if len(test_y) == 0:
                continue

            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)

            # Save to object
            predicted_ratings[current_id] = predictions
            true_ratings[current_id] = test_y

            # Save to vector
            predicted_ratings_vector.extend(predictions)
            true_ratings_vector.extend(test_y)

        # Print classification accuracy
        ca = accuracy_score(true_ratings_vector, predicted_ratings_vector)
        print("Classification Accuracy: " + str(ca))

        # Plot confusion matrix
        cm = confusion_matrix(true_ratings_vector, predicted_ratings_vector, labels=unique_ratings)
        df_cm = pd.DataFrame(cm, index=unique_ratings, columns=unique_ratings)
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt='d', annot_kws={"size": 22})
        plt.title('All images: KNN', fontsize=16)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        plt.show()
