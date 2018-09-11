#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np
import scipy.special as ssp


def drug_projection(n_latent: int, scale: Union[int, float], sparsity: float):
    """Generates a linear weighting from a latent space to feature space,
    potentially with some coefficients set to zero. Returns the weighting.

    :param n_latent: dimensionality of the latent space
    :param scale: scaling factor
    :param sparsity: relative sparsity of the weighting vector
    :return: array of shape (n_latent,) with weighting
    """

    n_nz_features = int(sparsity * n_latent)

    z_weights = np.zeros(n_latent)
    nz_i = np.random.choice(n_latent, n_nz_features, False)
    z_weights[nz_i] = scale * np.random.normal(size=n_nz_features)

    return z_weights


def drug_doses(n_latent: int, scale: Union[int, float], n_conditions: int):
    """
    Generates an array of uniformly-spaced values with a bit of random noise
    added in. Scaled to cover the expected range of the drug response data,

    :param n_latent: dimensionality of the relevant latent space
    :param scale: scaling factor used for z weights
    :param n_conditions: number of conditions (doses) desired
    :return: array shape (n_conditions,) with thresholds
    """

    # expected scale of the dot product Xz
    prod_scale = np.sqrt(n_latent * scale ** 2)

    dose_thresholds = (
        np.linspace(-3 * prod_scale, 3 * prod_scale, n_conditions)
        + np.random.normal(size=n_conditions, scale=1.0 / (n_conditions ** 2))
    )

    return dose_thresholds


def drug_response(X: np.ndarray, z_weights: np.ndarray, doses: np.ndarray):
    """Given an array of samples from a latent space, the weighting for a drug,
    and the dose thresholds, this calculates the expected outcome for each
    sample according to a logistic function

    :param X: array of samples with shape (n_samples, n_latent)
    :param z_weights: (n_latent,) weights for projection into the drug space
    :param doses: (n_conditions,) array of dose thresholds
    :return: (n_samples, n_conditions) array of outcomes
    """

    return ssp.expit(np.dot(X, z_weights)[..., None] + doses)
