import numpy as np
import matplotlib.pyplot as plt
from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix


def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """
    # TODO: Compute pairwise distances
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    square = diff ** 2
    affinity = square.sum(axis=2)
    if affinity_type == 'knn':
        # TODO: Find k nearest neighbors for each point

        # TODO: Construct symmetric affinity matrix

        # TODO: Return affinity matrix
        # indic = np.argsort(affinity, axis=1)[:, :k]
        # row_indices = np.arange(affinity.shape[0])[:, None]
        # A = np.zeros_like(affinity)
        # A[row_indices, indic] = 1
        n_samples = data.shape[0]

        # Step 1: Compute pairwise distances
        diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        # Step 2: Find k-nearest neighbors for each point
        # argsort each row and get the indices of the k smallest distances
        # Note: We include the point itself, which is always at position 0
        knn_indices = np.argsort(distances, axis=1)[:, :k + 1]

        # Step 3: Construct an affinity matrix based on k-nearest neighbors
        affinity_matrix = np.zeros((n_samples, n_samples))

        # Determine sigma if not provided
        if sigma is None:
            sigma = np.mean(distances)

        # Populate the affinity matrix
        for i in range(n_samples):
            for j in knn_indices[i]:
                # Gaussian kernel for similarity
                affinity_matrix[i, j] = np.exp(-distances[i, j] ** 2 / (2. * sigma ** 2))

        # Step 4: Make the affinity matrix symmetric
        affinity_matrix = np.maximum(affinity_matrix, affinity_matrix.T)
        A = affinity_matrix
    elif affinity_type == 'rbf':
        # TODO: Apply RBF kernel

        # TODO: Return affinity matrix
        A = np.exp(-affinity / (2 * sigma ** 2))
    else:
        raise Exception("invalid affinity matrix type")

    return A


if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']

    # TODO: Create and configure plot
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
    counter = 0
    for ds_name in datasets:
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset['data']     # feature points
        y = dataset['target']   # ground truth labels
        n = len(np.unique(y))   # number of clusters
        k = 3
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        # TODO: Create subplots
        axs[counter, 0].scatter(X[:, 0], X[:, 1], c=y)
        axs[counter, 0].set_title(f'Ground Truth: {ds_name}')

        # Plot results of K-means
        axs[counter, 1].scatter(X[:, 0], X[:, 1], c=y_km)
        axs[counter, 1].set_title(f'K-means on {ds_name}')

        # Plot results of Spectral Clustering with RBF
        axs[counter, 2].scatter(X[:, 0], X[:, 1], c=y_rbf)
        axs[counter, 2].set_title(f'RBF affinity on {ds_name}')

        # Plot results of Spectral Clustering with KNN
        axs[counter, 3].scatter(X[:, 0], X[:, 1], c=y_knn)
        axs[counter, 3].set_title(f'KNN affinity on {ds_name}')
        counter += 1
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    # TODO: Show subplots
