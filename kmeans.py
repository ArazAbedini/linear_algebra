import numpy as np



def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means_clustering(data, k, max_iterations=100):
    """
    Perform K-means clustering on the given dataset.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - k: int, the number of resulting clusters.
    - max_iterations: int, optional parameter to prevent potential infinite loops (default: 100).

    Returns:
    - labels: numpy array, cluster labels for each data point.
    - centroids: numpy array, final centroids of the clusters.
    """

    # TODO: Randomly initialize centroids
    
    # TODO: Iterate until convergence and update centroids and labels
    
    # TODO: Return labels and centroids

    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        old_centroids = centroids
        labels = assign_clusters(data, centroids)
        centroids = update_centroids(data, labels, k)
        if np.all(centroids == old_centroids):
            break

    return labels, centroids
