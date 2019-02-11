#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, Tuple, Union

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
                  same as for ``sparsity``
    :return: array of shape (n_rows, n_cols)
    """
    if np.any(sparsity <= 0) or np.any(sparsity > 1):
        raise ValueError(f"Sparsity should be in the interval (0, 1]")

    # fmt: off
    weights = (
        (np.random.random(size=(n_rows, n_cols)) < sparsity)
        * np.random.normal(loc=0.0, scale=scale, size=(n_rows, n_cols))
    )
    # fmt: on

    if np.any((weights != 0).sum(0) == 0):
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
                  ``n_latent`` is given, sets the scale per program
    :return: array of shape (n_latent, n_features)
    """
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
                     average. An array of size ``(n_latent,)`` sets the rate
                     per program
    :param scale: scaling factor for program weighting. If an array
                  of size ``(n_latent,)`` is given, different values are used
                  for each of the programs
    :return: array of shape (n_classes, n_latent)
    """
    # broadcast to correct dimensions
    sparsity = np.broadcast_to(sparsity, (n_latent,))
    scale = np.broadcast_to(scale, (n_latent,))

    classes = gen_weighting(n_classes, n_latent, sparsity, scale)

    return classes


def gen_class_samples(
    n_samples: int, class_weighting: np.ndarray, cov: np.ndarray = None
) -> np.ndarray:
    """Given a class weighting on a latent space and a number of cells,
    produce a sample of cells from the given class. Cells have random
    noise added along the dimensions specified by their class programs

    :param n_samples: number of samples (cells) to generate
    :param class_weighting: the class weightings on a latent space
    :param cov: covariance matrix for noise in the latent space
    :return: array of (n_samples, n_latent) observations
    """
    n_latent = class_weighting.shape[0]
    if cov is None:
        cov = np.eye(n_latent)

    class_programs = np.diagflat(class_weighting != 0).astype(int)

    z_noise = np.random.multivariate_normal(
        mean=np.zeros(n_latent), cov=cov, size=n_samples
    )

    return class_weighting + np.dot(z_noise, class_programs)


def sample_classes(
    n_samples: int,
    classes: np.ndarray,
    proportions: np.ndarray = None,
    cells_per_class: Union[int, np.ndarray] = None,
    program_cov: Union[np.ndarray, Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given the class weightings on the latent space and a number of cells,
    produce a sample of cells based on the given class proportions. Cells have
    random noise added along the dimensions specified by their class programs

    :param n_samples: number of samples (cells) to generate
    :param classes: the class weightings on a latent space
    :param proportions: proportions for each class. Mutually exclusive with
                        `cells_per_class`
    :param cells_per_class: counts for each class, either as a constant or per-class.
                            Mutually exclusive with `proportions`
    :param program_cov: covariance matrix, or a callable that generates matrices.
                        If None, the identity matrix is used.
    :return: array of (n_samples, n_latent) observations and (n_samples,) class labels
    """

    n_classes, n_latent = classes.shape

    if (proportions is None) == (cells_per_class is None):
        raise ValueError(
            "Either `proportions` or `cells_per_class` must be specified, but not both."
        )
    elif proportions is not None:
        labels = np.random.choice(n_classes, n_samples, p=proportions)
    else:
        labels = np.random.permutation(np.repeat(np.arange(n_classes), cells_per_class))

    if isinstance(program_cov, Callable):
        cov_gen = program_cov
    else:
        cov_gen = lambda: program_cov

    sample_z = np.empty((n_samples, n_latent))

    for i, n_i in zip(*np.unique(labels, return_counts=True)):
        sample_z[labels == i, :] = gen_class_samples(n_i, classes[i, :], cov_gen())

    return sample_z, labels
