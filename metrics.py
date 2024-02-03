import numpy as np



def entropy(labels):
    n = len(labels)
    label_counts = np.unique(labels, return_counts=True)[1]
    H = -np.sum((label_counts / n) * np.log(label_counts / n))
    return H


def mutual_information(U, V):
    n = len(U)
    U_labels, U_counts = np.unique(U, return_counts=True)
    V_labels, V_counts = np.unique(V, return_counts=True)
    MI = 0
    for i in U_labels:
        for j in V_labels:
            ij_intersection = np.logical_and(U == i, V == j).sum()
            if ij_intersection > 0:
                P_ij = ij_intersection / n
                P_i = U_counts[U_labels == i][0] / n
                P_j = V_counts[V_labels == j][0] / n
                MI += P_ij * np.log(P_ij / (P_i * P_j))
    return MI

def normalized_mutual_information(U, V):
    return 2 * mutual_information(U, V) / (entropy(U) + entropy(V))


def clustering_score(true_labels, predicted_labels):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """

    # TODO: Calculate and return clustering score

    nmi = normalized_mutual_information(true_labels, predicted_labels)
    return nmi
