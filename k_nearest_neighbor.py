import numpy as np


class KNearestNeighbor(object):
    """A kNN classifier with L2 distance."""

    def __init__(self):
        """Initialize the classifier with no training data yet."""
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Memorize the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data.
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError(
                'Invalid value %d for num_loops' % num_loops
            )

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute Euclidean distances using two loops.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(
                    np.sum((X[i] - self.X_train[j]) ** 2)
                )

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute Euclidean distances using one loop.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            diff = self.X_train - X[i]
            dists[i, :] = np.sqrt(
                np.sum(diff ** 2, axis=1)
            )

        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute Euclidean distances without explicit loops.
        """
        x2 = np.sum(X ** 2, axis=1, keepdims=True)
        y2 = np.sum(
            self.X_train ** 2, axis=1, keepdims=True
        ).T

        xy = X.dot(self.X_train.T)

        dists_squared = x2 + y2 - 2 * xy
        dists_squared = np.maximum(dists_squared, 0)

        dists = np.sqrt(dists_squared)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Predict labels using majority voting among k nearest neighbors.
        """
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError('k must be a positive integer')

        num_test = dists.shape[0]
        num_train = self.y_train.shape[0]

        if num_train == 0:
            raise ValueError('Training data is empty')

        k_eff = min(k, num_train)
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            closest_idx = np.argsort(dists[i])[:k_eff]
            closest_y = self.y_train[closest_idx]

            labels, counts = np.unique(
                closest_y, return_counts=True
            )

            y_pred[i] = labels[np.argmax(counts)]

        return y_pred
