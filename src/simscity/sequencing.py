#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union

import numpy as np
import scipy.stats as st


def library_size(
    n_cells: int,
    loc: float = 7.5,
    scale: float = 0.5,
    lower_bound: float = -1.0,
    upper_bound: float = np.inf,
) -> np.ndarray:
    """log-normal noise for the number of umis per cell (with a lower bound to
    represent a minimum depth cutoff)

    :param n_cells: number of library sizes to generate
    :param loc: mean of library size in log-space
    :param scale: standard deviation
    :param lower_bound: lower bound relative to ``loc``
    :param upper_bound: upper bound relative to ``loc``
    :return: float array of shape (n_cells,) containing the library sizes
    """

    return np.exp(
        st.truncnorm.rvs(lower_bound, upper_bound, loc=loc, scale=scale, size=n_cells)
    ).astype(int)


def fragment_genes(n_genes: int, lam: float = 1.0) -> np.ndarray:
    """Generate a random number of fragments for each gene used a poisson distribution

    :param n_genes: number of genes to fragment
    :param lam: mean of poisson distribution of (additional) fragments. Every gene
                will have at least one fragment
    :return: int array of shape (n_genes,) with number of fragments per gene
    """
    # random number of possible fragments per gene, poisson distributed
    # add one to ensure ≥1 fragment per gene
    fragments_per_gene = 1 + np.random.poisson(lam, size=n_genes)

    return fragments_per_gene


def umi_counts(
    raw_expression: np.ndarray,
    lib_size: Union[int, np.ndarray] = None,
    fragments_per_gene: Union[int, np.ndarray] = 1,
) -> np.ndarray:
    """Given an (n_samples, n_genes) array of expression values, generates
    a count matrix of UMIs based by multinomial sampling

    :param raw_expression: array of raw expression values (non-negative)
    :param lib_size: library size for each cell, either constant or per-sample.
                     If None, generates a distribution using ``library_size``
    :param fragments_per_gene: fragments observed per gene, either constant or per-gene
    :return: integer array of shape (n_samples, n_features) containing umi counts
    """

    if np.any(raw_expression < 0):
        raise ValueError("raw_expression must be non-negative")

    n_cells, n_genes = raw_expression.shape

    if lib_size is None:
        lib_size = library_size(n_cells)  # is this a good default?
    else:
        lib_size = np.broadcast_to(lib_size, (n_cells,))

    fragments_per_gene = np.broadcast_to(fragments_per_gene, (n_genes,))

    # each fragment is at the level of the gene it comes from
    fragment_expression = np.repeat(raw_expression, fragments_per_gene, axis=1)

    gene_p = fragment_expression / fragment_expression.sum(1, keepdims=True)

    cell_gene_umis = np.vstack(
        [
            np.random.multinomial(n=lib_size[i], pvals=gene_p[i, :])
            for i in range(n_cells)
        ]
    )

    return cell_gene_umis


def pcr_noise(
    read_counts: np.ndarray,
    pcr_betas: Union[float, np.ndarray],
    n_cycles: int,
    copy: bool = True,
) -> np.ndarray:
    """PCR noise model: every read has an affinity for PCR, and for every round
    of PCR we do a ~binomial doubling of each count.

    :param read_counts: array of shape (n_samples, n_features) representing unique
                        molecules (e.g. genes or gene fragments)
    :param pcr_betas: PCR efficiency for each feature, either constant or per-feature
    :param n_cycles: number of rounds of PCR to simulate
    :param copy: return a copy of the read_counts array or modify in-place
    :return: int array of shape (n_samples, n_features) with amplified counts
    """
    if np.any(pcr_betas < 0):
        raise ValueError("pcr_betas must be non-negative")

    if copy:
        read_counts = read_counts.copy()

    pcr_betas = np.broadcast_to(pcr_betas, (1, read_counts.shape[1]))

    # for each round of pcr, each gene increases according to its affinity factor
    for i in range(n_cycles):
        read_counts += np.random.binomial(
            n=read_counts, p=pcr_betas, size=read_counts.shape
        )

    return read_counts
