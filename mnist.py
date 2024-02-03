import numpy as np
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two point clouds.

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - dist: float, the Chamfer distance between the two point clouds.
    """
    
    # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2

    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1

    # TODO: Return Chamfer distance, sum of the average distances in both directions

    pass

def register(point_cloud1, point_cloud2):
    """
    Registers point_cloud1 and point_cloud2 to align them and optimize distance

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - registered_point_cloud1: numpy array, shape (N1, D), representing transformed version of the first point cloud.
    - registered_point_cloud2: numpy array, shape (N2, D), representing transformed version of the second point cloud.
    """

    # TODO: Find a rigid or non-rigid transformation

    # TODO: Transform point clouds by transformation

    # TODO: Return transformed point clouds

    pass

def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    # TODO: Iterate over point clouds to fill affinity matrix

    # TODO: For each pair of point clouds, register them with each other

    # TODO: Calculate symmetric Chamfer distance

    pass


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Plot Ach using its first 3 eigenvectors
