
from typing import Union

import numpy as np
import pandas
import scanpy


def latent_space(n_latent: int = 8, n_classes: int = 3,
                 scale: Union[int, float] = 5.0):
    """Generates the centroids of n_classes in a latent space

    :param n_latent: dimensionality of the latent space
    :param n_classes: number of different classes
    :param scale: scaling factor
    :return: array of shape (n_classes, n_latent)
    """

    if n_latent == 2:
        # evenly spaced on the unit circle
        class_centers = scale * np.array(
                [[np.cos(2 * np.pi / n_classes * i),
                  np.sin(2 * np.pi / n_classes * i)]
                 for i in range(n_classes)]
        )
    elif n_latent > 2:
        # randomly generated from i.i.d. standard normal of n_latent dim
        x = np.random.normal(0, 1, size=(n_classes, n_latent))
        class_centers = scale * x / np.linalg.norm(axis=1)[:, None]
    else:
        raise ValueError(f'n_latent = {n_latent} is not supported')

    return class_centers


def sample_classes(class_centers: np.ndarray, n_obs: int,
                   proportions: np.ndarray):
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



def gen_batch_vectors(W: np.ndarray, n_batches: int, n_features: int,
                      batch_scale: float, bio_batch_angle: float):
    """Generates a batch-effect vector for each batch

    :param W: weighting that translate from latent to feature space
    :param n_batches: number of batches
    :param n_features: number of features
    :param batch_scale: size of batch effect relative to data
    :param bio_batch_angle: angle of batch effect relative to biology
    :return: array of shape (n_batches, n_features)
    """
    def norm(X):
        return np.linalg.norm(X, axis=1, keepdims=True)

    batch_vectors = np.random.randn(n_batches, n_features)
    batch_vectors = (batch_vectors / norm(batch_vectors)
                     * np.mean(norm(expression)) * batch_scale)

    # The col-span of W is the bio-space
    if bio_batch_angle is not None:
        projection_to_bio = np.dot(np.linalg.pinv(W), W)

        v_projected = np.dot(batch_vectors, projection_to_bio)
        v_complement = batch_vectors - v_projected

        batch_vectors = norm(batch_vectors) * (
                np.sin(bio_batch_angle) * v_complement / norm(v_complement) +
                np.cos(bio_batch_angle) * v_projected / norm(v_projected)
        )

    return batch_vectors


def true_expression(n_obs: int = 1000, n_features: int = 100,
                    n_batches: int = 2, n_latent: int = 2, n_classes: int = 3,
                    proportions: np.ndarray = None, seed: int = 2018,
                    scale: Union[int, float]=5, batch_scale: float = 0.1,
                    bio_batch_angle: float = None):
    """

    :param n_obs: number of observations (cells) per batch
    :param n_features: number of features (genes)
    :param n_batches: number of batches
    :param n_latent: size of the latent space used to generate data
    :param n_classes: number of classes shared across batches
    :param proportions: proportion of cells from each class in each batch
                        default is equal representation
    :param seed: seed for random number generator
    :param scale: scaling factor for generating data
    :param batch_scale: batch effect relative to data
    :param bio_batch_angle: angle of batch effect w/ bio subspace
    :return:
    """

    if proportions is None:
        proportions = np.ones((n_batches, n_classes)) / n_classes
    else:
        proportions = np.asarray(proportions)

    assert (n_batches, n_classes) == proportions.shape

    if seed:
        np.random.seed(seed)

    class_centers = latent_space(n_latent, n_classes, scale)

    latent = np.zeros((n_batches * n_obs, n_latent))
    metadata = pd.DataFrame({'batch': np.repeat(range(n_batches), n_obs),
                             'class': [-1] * (n_obs * n_batches)})

    W = np.random.randn(n_latent, n_features)

    for batch in range(n_batches):
        id_range = np.arange(batch * n_obs, ((batch + 1) * n_obs))

        obs, classes = sample_classes(class_centers, n_obs, proportions[batch, :])

        latent[id_range, :] = obs
        metadata.iloc[id_range, 1] = classes


    expression = np.dot(latent, W)
    expression_gt = expression.copy()

    # add batch vector
    batch_vectors = gen_batch_vectors(W, n_batches, n_features,
                                      batch_scale, bio_batch_angle)

    for batch in range(n_batches):
        expression[metadata['batch'] == batch, :] += batch_vectors[batch, :]


    metadata['batch'] = metadata['batch'].astype('category')
    metadata['class'] = metadata['class'].astype('category')

    adata = scanpy.AnnData(X=expression, obs=metadata,
                           obsm={'X_latent': latent, 'X_gt': expression_gt})

    return adata
