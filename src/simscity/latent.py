#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Tuple, Union

import numpy as np

import warnings


def gen_weighting(
    n_rows: int, n_cols: int, sparsity: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Generic function to generate a random weighting. Can be used to create
    a random projection from one space to another

    :param n_rows: dimensionality of input space
    :param n_cols: dimensionality of output space
    :param sparsity: probability that a weight is nonzero. If shape is (n_cols,)
                     or (1, n_cols), different values will apply to each columns.
                     If shape is (n_rows, 1) the values will apply to each row.
    :param scale: standard deviation of the weights. The rules for shape are the
                  same as for `sparsity`
    :return:
    """
    if np.any(sparsity <= 0):
        raise ValueError(f"Sparsity must be non-negative")

    # fmt: off
    weights = (
        (np.random.random(size=(n_rows, n_cols)) < sparsity)
        * np.random.normal(loc=0.0, scale=scale, size=(n_rows, n_cols))
    )
    # fmt: on

    if np.any((weights != 0).sum(1) == 0):
        warnings.warn(
            "Some columns have no nonzero weights. Consider increasing sparsity"
        )

    return weights


def gen_programs(
    n_latent: int,
    n_features: int,
    sparsity: Union[float, np.ndarray],
    scale: Union[float, np.ndarray],
) -> np.ndarray:
    """Generate different "programs", each of which consists of a weighting
    across the n_features of the biological space.

    :param n_latent: dimensionality of the latent space (number of programs)
    :param n_features: dimensionality of the feature space
    :param sparsity: probability that a feature is used by a given program. Each
                     feature will be selected `sparsity * n_latent` times on
                     average. An array of size `(n_features,)` sets the rate per
                     feature
    :param scale: scaling factor for feature weighting. If an array of size
                  `n_latent` is given, sets the scale per program
    :return: array of shape (n_latent, n_features)
    """
    if isinstance(sparsity, float) and sparsity >= 1.0:
        warnings.warn(
            f"Sparsity {sparsity} >= 1.0, every feature will be used by every program"
        )

    # broadcast to correct dimensions
    sparsity = np.broadcast_to(sparsity, (n_features,))
    scale = np.broadcast_to(scale, (n_latent, 1))

    programs = gen_weighting(n_latent, n_features, sparsity, scale)

    return programs


def gen_classes(
    n_latent: int,
    n_classes: int,
    sparsity: Union[float, np.ndarray],
    scale: Union[float, np.ndarray],
) -> np.ndarray:
    """Generates the *program weights* for n_classes in a latent space. Each
    class is made up of a random selection of the available programs based on
    the given sparsity

    :param n_latent: dimensionality of the latent space
    :param n_classes: number of different classes
    :param sparsity: probability that a program is used by a given class. Each
                     program will be selected `sparsity * n_classes` times on
                     average. An array of size `(n_latent,)` sets the rate per
                     program
    :param scale: scaling factor for program weighting. If an array
                  of size `(n_latent,)` is given, different values are used for
                  each of the programs
    :return: array of shape (n_classes, n_latent)
    """

    if isinstance(sparsity, float) and sparsity >= 1.0:
        warnings.warn(
            f"Sparsity {sparsity} >= 1.0, every program will be used by every class"
        )

    # broadcast to correct dimensions
    sparsity = np.broadcast_to(sparsity, (n_latent,))
    scale = np.broadcast_to(scale, (n_latent,))

    classes = gen_weighting(n_classes, n_latent, sparsity, scale)

    return classes


def sample_classes(
    n_obs: int, classes: np.ndarray, proportions: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Given the class weightings on the latent space and a number of cells,
    produce a sample of cells based on the given class proportions. Cells have
    random noise added along the dimensions specified by their class programs

    :param n_obs: number of observations (cells) to generate
    :param classes: the class weightings on a latent space
    :param proportions: proportions for each class
    :return: array of shape (n_obs, n_latent) observations and (n_obs,) class labels
    """
    n_classes, n_latent = classes.shape

    if proportions is None:
        proportions = np.ones(n_classes) / n_classes
    else:
        proportions = np.asarray(proportions)

    labels = np.random.choice(n_classes, n_obs, p=proportions)

    class_programs = [
        np.diagflat(classes[i, :] != 0).astype(int) for i in range(n_classes)
    ]

    z_noise = np.random.standard_normal((n_obs, n_latent))
    obs_z = classes[labels, :]

    for i in range(n_classes):
        obs_z[labels == i, :] += np.dot(z_noise[labels == i, :], class_programs[i])

    return obs_z, labels
