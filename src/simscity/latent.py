
from typing import Union

import numpy as np
import scipy as st


def gen_classes(n_latent: int = 8, n_classes: int = 3,
                scale: Union[int, float] = 5.0, random: bool = False):
    """Generates the centroids of n_classes in a latent space

    :param n_latent: dimensionality of the latent space
    :param n_classes: number of different classes
    :param scale: scaling factor
    :param random: generate random vectors vs try to evenly space them. If
                   `n_latent = 2` uses unit circle, otherwise unit hypercube
    :return: array of shape (n_classes, n_latent) where each class vector has
             length equal to `scale`
    """

    if random or n_classes > 2**n_latent:
        if n_classes:
            raise RuntimeWarning('Too many classes for unit cube, randomizing')

        x = np.random.normal(0, 1, size=(n_classes, n_latent))
    elif n_latent == 2:
        # evenly spaced on the unit circle
        x = np.array([[np.cos(2 * np.pi / n_classes * i),
                       np.sin(2 * np.pi / n_classes * i)]
                      for i in range(n_classes)])
    else:
        # vertices of the n_latent-dimensional hypercube
        i2v = lambda i: list(map(int, bin(i)[2:].zfill(n_latent)))

        x = np.array([(-1) ** np.array(i2v(i))
                      for i in range(n_classes)]).astype(float)

    class_centers = scale * x / np.linalg.norm(x, axis=1, keepdims=True)

    return class_centers


def gen_projection(n_latent: int, n_features: int):
    """Generates a linear weighting from a latent space to feature space,
    and returns that project and it's inverse

    :param n_latent: dimensionality of the latent space
    :param n_features: dimensionality of the feature space
    :return: array of shape (n_latent, n_features) with weighting, and array
             of shape (n_features, n_features) projecting on to the
    """

    z_weights = np.random.randn(n_latent, n_features)
    projection_to_bio = np.dot(np.linalg.pinv(z_weights), z_weights)

    return z_weights, projection_to_bio


def sample_classes(class_centers: np.ndarray, n_obs: int, proportions: np.ndarray):
    """Given the class centroids in latent space and a number of cells, produce
    a sample of cells based on the given proportions

    :param class_centers: the class centroids in a latent space
    :param n_obs: number of cells to generate
    :param proportions: proportions for each class
    :return: array of shape (n_obs, n_latent) and their class labels
    """
    classes = np.random.choice(class_centers.shape[0], n_obs, p=proportions)

    centers = class_centers[classes, :]
    spread = np.random.randn(n_obs, class_centers.shape[1])

    return centers + spread, classes


def latent_expression(n_obs: int, class_centers: np.ndarray,
                      proportions: np.ndarray = None):
    """Sample a batch from a set of class centroids based on a vector of proportions

    :param n_obs: number of observations (cells) to sample
    :param class_centers: centroids of the dfiferent classes, in latent space
    :param proportions: proportion of cells from each class in the batch
                        default is equal representation
    :return: latent array (n_obs, n_latent) and class labels
    """

    n_classes, n_latent = class_centers.shape

    if proportions is None:
        proportions = np.ones(n_classes) / n_classes
    else:
        proportions = np.asarray(proportions)

    assert proportions.shape[0] == n_classes

    obs_z, classes = sample_classes(class_centers, n_obs, proportions)

    return obs_z, classes
