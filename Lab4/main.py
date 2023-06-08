import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from LAb_4 import centroid_neural_net


def plot_cnn_result(input_data, centroids, cluster_indices, figure_size=(8, 8)):
    X = input_data
    num_clusters = len(centroids)

    plt.figure(figsize=figure_size)

    cnn_cluster_elements = []

    for i in range(num_clusters):
        display = []
        for x_th in range(len(X)):
            if cluster_indices[x_th] == i:
                display.append(X[x_th])

        cnn_cluster_elements.append(display)

        display = np.array(display)
        plt.scatter(display[:, 0], display[:, 1])
        plt.scatter(centroids[i][0], centroids[i][1], s=200, c='red')
        plt.text(centroids[i][0], centroids[i][1], f"Cluster {i}", fontsize=14)
    plt.show()


data = np.loadtxt('data2.txt')

# Extract the x and y coordinates from the data
X = data

# Create a scatter plot of the data
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

num_clusters = 5
centroids, w, cluster_indices, cluster_elements, cluster_lengths, epochs = centroid_neural_net(X, num_clusters,
                                                                                               max_iteration=1000,
                                                                                               epsilon=0.05)
plot_cnn_result(X, centroids, cluster_indices, figure_size=(8, 8))
