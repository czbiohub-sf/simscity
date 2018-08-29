#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np
import scipy.stats as st


def library_size(n_cells: int, loc: float = 8.5, scale: float = 1.5,
                 lower_bound: float = -1.0, upper_bound: float = np.inf):
    """log-normal noise for the number of reads (with a lower bound to
    represent a minimum depth cutoff)

    :param n_cells: number of library sizes to generate
    :param loc: mean of library size in log-space
    :param scale: standard deviation
    :param lower_bound: lower bound relative to `loc`
    :param upper_bound: upper bound relative to `loc`
    :return:
    """

    return np.exp(st.truncnorm.rvs(
        lower_bound, upper_bound, loc=loc, scale=scale, size=n_cells
    )).astype(int)


def umi_counts(raw_expression: np.ndarray, lib_size: np.ndarray = None):
    """Given an (n_obs, n_features) array of true expression values, generates
    a count matrix of UMIs based by multinomial sampling with a variable-sized
    library of reads using a truncated log-normal distribution

    :param raw_expression: expression of true expression values
    :param lib_size: library size for each cell
    :return: int array of shape (n_obs, n_features) containing umi counts
    """

    n_cells, n_genes = raw_expression.shape

    if lib_size is None:
        lib_size = library_size(n_cells)

    gene_p = raw_expression / raw_expression.sum(1)[:, None]

    cell_gene_umis = np.vstack(
        [np.random.multinomial(n=lib_size[i], pvals=gene_p[i, :])
         for i in range(n_cells)]
    )

    return cell_gene_umis


def pcr_noise(read_counts: np.ndarray, pcr_betas: np.ndarray, n: int,
              copy: bool = True):
    """PCR noise model: every read has an affinity for PCR, and for every round
    of PCR we do a ~binomial doubling of each count.

    :param read_counts: array of shape (n_cells, n_genes) representing unique
                        molecules
    :param pcr_betas: read-specific PCR efficiencies of shape (n_genes,)
    :param n: number of rounds of PCR to simulate
    :param copy: return a copy of the read_counts array or modify in-place
    :return: int array of shape (n_cells, n_genes) with amplified counts
    """
    if copy:
        read_counts = read_counts.copy()

    # for each round of pcr, each gene increases according to its affinity factor
    for i in range(n):
        read_counts += np.random.binomial(read_counts, pcr_betas[None, :],
                                          size=read_counts.shape)

    return read_counts
