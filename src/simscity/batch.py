#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np


def gen_batch_vectors(
    n_batches: int,
    n_features: int,
    batch_scale: float,
    bio_batch_angle: Union[float, None] = None,
    projection_to_bio: Union[np.ndarray, None] = None,
):
    """Generates a batch-effect vector for each batch, optionally with some
    relation to the biological space

    :param n_batches: number of batches
    :param n_features: number of features
    :param batch_scale: size of batch effect relative to data
    :param bio_batch_angle: angle of batch effect w/ bio subspace
    :param projection_to_bio: projection from latent into gene space
    :return: array of shape (n_batches, n_features)
    :rtype: np.ndarray
    """

    def norm(X):
        return np.linalg.norm(X, axis=1, keepdims=True)

    batch_vectors = np.random.randn(n_batches, n_features)
    batch_vectors = (
        batch_vectors / norm(batch_vectors) * np.mean(norm(expression)) * batch_scale
    )

    if bio_batch_angle is not None:
        v_projected = np.dot(batch_vectors, projection_to_bio)
        v_complement = batch_vectors - v_projected

        batch_vectors = norm(batch_vectors) * (
            np.sin(bio_batch_angle) * v_complement / norm(v_complement)
            + np.cos(bio_batch_angle) * v_projected / norm(v_projected)
        )

    return batch_vectors


def add_batch_vectors(
    expression: np.ndarray,
    batch: np.ndarray,
    batch_scale: Union[int, float],
    bio_batch_angle: Union[float, None],
    projection_to_bio: Union[np.ndarray, None],
    copy: bool = True,
):
    """Generate batch-effect vectors and apply them to the expression data

    :param expression: array of true expression, in latent space
    :param batch: indicator of which obs belongs to which batch
    :param batch_scale: batch effect relative to data
    :param bio_batch_angle: angle of batch effect w/ bio subspace
    :param projection_to_bio: projection from latent into gene space
    :param copy: return a copy of the expression array or modify in-place
    :return: expression matrix with batch effect
    :rtype: np.ndarray
    """
    if copy:
        expression = expression.copy()

    n_batches = len(np.unique(batch))

    # add batch vector
    batch_vectors = gen_batch_vectors(
        n_batches, expression.shape[1], batch_scale, bio_batch_angle, projection_to_bio
    )

    for i in range(n_batches):
        expression[batch == i, :] += batch_vectors[i, :]

    return expression
