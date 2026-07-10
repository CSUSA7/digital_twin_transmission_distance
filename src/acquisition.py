"""
Acquisition functions for Bayesian optimization, shared between the teaching
notebook (03) and the beamline emulator notebook (05) so they are not
duplicated in two places.
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(mean, std, best, xi=0.01):
    """Expected improvement for minimization. Returns one score per point."""
    mean = np.ravel(mean)
    std = np.ravel(std)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (best - mean - xi) / std
        ei = (best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std == 0, 0.0, ei)
    return ei


def lower_confidence_bound(mean, std, beta=2.0):
    """Score for minimization: rewards a low mean and a large std."""
    return -(np.ravel(mean) - beta * np.ravel(std))
