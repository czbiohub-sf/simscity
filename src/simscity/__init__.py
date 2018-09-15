#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np

from simscity import batch, drug, latent, sequencing, util


def mnn_synthetic_data(
    n_obs: int = 1000,
    n_features: int = 100,
    n_batches: int = 2,
    n_latent: int = 2,
    n_classes: int = 3,
    proportions: np.ndarray = None,
    sparsity: float = 1.0,
    scale: Union[int, float] = 5,
    batch_scale: float = 0.1,
    bio_batch_angle: Union[float, None] = None,
    seed: int = 2018,
):
    """	
    :param n_obs: number of observations (cells) per batch
    :param n_features: number of features (genes)	
    :param n_batches: number of batches	
    :param n_latent: size of the latent space used to generate data	
    :param n_classes: number of classes shared across batches	
    :param proportions: proportion of cells from each class in each batch. If
                        shape is (n_classes,) same proportions used each time.
                        If shape is (n_batches, n_classes) then each row is a
                        different batch. Default is equal representation
    :param sparsity: sparsity of class weightings
    :param scale: scaling factor for generating data
    :param batch_scale: batch effect relative to data
    :param bio_batch_angle: angle of batch effect w/ bio subspace
    :param seed: seed for random number generator
    :return: real-valued expression data with batch effect and metadata
    """

    if proportions is None:
        proportions = np.ones((n_batches, n_classes)) / n_classes
    else:
        proportions = np.broadcast_to(proportions, (n_batches, n_classes))

    if seed:
        np.random.seed(seed)

    class_centers = latent.gen_classes(n_latent, n_classes, sparsity, scale)

    batches = np.repeat(np.arange(n_batches), n_obs)
    latent_exp = []
    classes = []

    for b in range(n_batches):
        b_latent, b_classes = latent.sample_classes(
            n_obs, class_centers, proportions[b, :]
        )
        latent_exp.append(b_latent)
        classes.append(b_classes)

    latent_exp = np.vstack(latent_exp)
    classes = np.hstack(classes)

    programs = latent.gen_programs(n_latent, n_features, 1.0, 1.0)

    expression = np.dot(latent_exp, programs)

    projection_to_bio = np.dot(np.linalg.pinv(programs), programs)

    expression_w_batch = batch.add_batch_vectors(
        expression, batches, batch_scale, bio_batch_angle, projection_to_bio, copy=True
    )

    adata = util.arrays_to_anndata(
        expression_w_batch, batches, classes, X_latent=latent_exp, X_gt=expression
    )

    return adata
