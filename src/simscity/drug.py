#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np
import scipy.stats as st


# x_noise_scale = 2.0 # std dev of noise for data generation
# w_n = 2 # number of components to generate for randomzied covariance matrix
#
# n_classes = 64 # number of classes (cell lines)
# n_samples_per_class = 256 # number of samples per class (single cells per line)
# n_features = 1024 # number of features
# n_nz_features = 16 # number of features with non-zero coefficients
# n_conditions = 12 # number of hopefully-monotonically-related conditions (doses)


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
    nz_i = np.random.choice(n_features, n_nz_features, False)
    z_weights[nz_i] = scale * np.random.normal(size=n_nz_features)

    return z_weights


#
# # per-cell drug resistance values
# true_y = np.dot(x_vals, true_coef)
#
# y_lo, y_hi = np.percentile(true_y, (5., 95.))
#
# # assume that our space of accessible drug doses overlaps with the space of cell responses
# # e.g. we have cells ranging from very sensitive to very resistant
# dose_space = np.linspace(y_lo, y_hi, n_conditions*100)
#
#
# # randomized stuff -- will overwrite
#
# # actual doses measured will be much smaller, and their locations are unknown
# dose_thresholds = np.linspace(y_lo, y_hi, n_conditions) + np.random.normal(scale=(y_hi - y_lo) / (n_conditions*10))
# print('dose_thresholds\n', '\t'.join('{:.2f}'.format(d) for d in dose_thresholds))
#
# drs = np.random.binomial(
#     n_samples_per_class, ssp.expit(true_y[...,None] + dose_thresholds)
# ).mean(1).astype(np.float32) / n_samples_per_class
#
# true_drs = np.random.binomial(
#     n_samples_per_class, ssp.expit(true_y[...,None] + dose_space)
# ).mean(1).astype(np.float32) / n_samples_per_class
