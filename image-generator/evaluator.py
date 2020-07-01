import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import calculate_features
import pickle


COLORS_RATINGS = {1: 'yellow', 2: 'orange', 3: 'red'}
COLORS_CATEGORIES = {'leisure': 'yellow', 'adventure': 'red', 'religious': 'green', 'business': 'blue', 'sport and recreation': 'pink', 'health or medical': 'purple', 'cultural': 'coral'}


class Evaluator:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def draw_features(self, features, show, visualization_algorithm, categories, ratings):
        """
        Draw features in 2D space using pca or tsne, color for categories or ratings
        """
        print('draw features')

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

    def visualize(self, show='ratings', visualization_algorithm='tsne'):
        file_names = os.listdir(self.data_directory)
        name_vector = [self.data_directory + '/' + name for name in file_names]
        categories_vector = [name.split("-")[1] for name in file_names]
        ratings_vector = [int(name.split(".")[0].split("-")[2]) for name in file_names]
        features_file = 'data/features-' + self.data_directory.split('/')[-1] + '-tmp'

        if not os.path.exists(features_file):
            features = calculate_features(name_vector)
            with open(features_file, 'wb') as f:
                pickle.dump(features, f)
        else:
            with open(features_file, 'rb') as f:
                features = pickle.load(f)

        self.draw_features(features, show, visualization_algorithm, categories_vector, ratings_vector)

    def classify(self, algorithm='knn'):
        pass
