from kmeans import k_means_clustering
from sklearn.cluster import KMeans
from numpy import linalg as LA
import numpy as np

def laplacian(A):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """

    # TODO: Calculate degree matrix

    # TODO: Calculate the inverse square root of the symmetric matrix

    # TODO: Return symmetric normalized Laplacian matrix
    # Calculate degree matrix D
    D = np.diag(np.sum(A, axis=1))
    # Calculate the inverse square root of the degree matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    # Calculate symmetric normalized Laplacian matrix
    L_sym = np.identity(A.shape[0]) - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)

    return L_sym

def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    # TODO: Compute Laplacian matrix

    # TODO: Compute the first k eigenvectors of the Laplacian matrix

    # TODO: Apply K-means clustering on the selected eigenvectors

    # TODO: Return cluster labels

    lap_matrix = laplacian(affinity)
    eigenvalues, eigenvectors = np.linalg.eigh(lap_matrix)
    indices = np.argsort(eigenvalues)[:k]  # Indices of smallest k eigenvalues
    k_eigenvectors = eigenvectors[:, indices]

    # Apply K-means clustering on the selected eigenvectors
    # Assuming k_means_clustering is implemented as per your project requirements
    # If not, use KMeans from sklearn as a fallback
    # labels, _ = k_means_clustering(k_eigenvectors, k)
    # labels, _ = k_means_clustering(k_eigenvectors, k)
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(k_eigenvectors)

    return labels
