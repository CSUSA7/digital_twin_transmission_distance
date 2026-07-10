"""
Acquisition and candidate generation for the distance emulator (notebook 06).

`acquisition.py` stays untouched; this module is what notebook 06 imports.
Everything here assumes MINIMIZATION of the modeled quantity (smaller
distance to the detector = better), which is the native direction of
expected improvement -- no sign flip needed, unlike the transmission
notebook (05), which had to negate the mean to maximize.
"""

import numpy as np
from scipy.stats import norm, qmc


def expected_improvement_min(mean, std, best, xi=0.01):
    """Expected improvement for minimization. Returns one score per point.

    Same formula as acquisition.expected_improvement, restated here so the
    distance notebook depends only on this module.
    """
    mean = np.ravel(mean)
    std = np.ravel(std)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (best - mean - xi) / std
        ei = (best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std == 0, 0.0, ei)
    return ei


def sobol_candidates(n, d, seed=None):
    """`n` scrambled-Sobol points in the d-dimensional unit cube.

    Space-filling beats iid-uniform noticeably in 8 dimensions: the same
    candidate budget leaves much smaller uncovered gaps. Sobol wants a
    power-of-two sample size, so we draw the next power of two and keep n.
    """
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 2))))
    return sampler.random_base2(m)[:n]


def local_candidates(centers, n_per_center=256, radius=0.05, rng=None):
    """Gaussian clouds around the given centers, clipped to the unit cube.

    A cheap trust-region flavor: the global Sobol sweep finds promising
    basins, and these local clouds let EI refine within them at a
    resolution the global sweep cannot afford in 8 dimensions.
    """
    rng = np.random.default_rng() if rng is None else rng
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    noise = rng.normal(scale=radius,
                       size=(len(centers), n_per_center, centers.shape[1]))
    cloud = (centers[:, None, :] + noise).reshape(-1, centers.shape[1])
    return np.clip(cloud, 0.0, 1.0)


def propose_next_point(gp, U_observed, y_observed, n_sobol=4096,
                       n_incumbents=5, n_local=256, radius=0.05,
                       xi=0.01, seed=None):
    """One expected-improvement step for MINIMIZING the modeled quantity.

    gp : model exposing .posterior(U) -> (mean, std), already fit in the
        same (warped) space as `y_observed`.
    U_observed, y_observed : points already evaluated (unit cube) and
        their targets in the gp's fitting space (log1p mm for DistanceGP).

    Candidates = global scrambled-Sobol sweep + Gaussian clouds around the
    `n_incumbents` best observed points. Returns (x_next, ei_value); the
    EI value is worth logging -- it says how much improvement the model
    still expects, and a collapse toward zero means the loop has converged
    (or the model is overconfident).
    """
    rng = np.random.default_rng(seed)
    incumbents = U_observed[np.argsort(np.ravel(y_observed))[:n_incumbents]]
    candidates = np.vstack([
        sobol_candidates(n_sobol, U_observed.shape[1], seed=rng),
        local_candidates(incumbents, n_per_center=n_local, radius=radius,
                         rng=rng),
    ])
    mean, std = gp.posterior(candidates)
    ei = expected_improvement_min(mean, std, best=float(np.min(y_observed)),
                                  xi=xi)
    best_idx = int(np.argmax(ei))
    return candidates[best_idx], float(ei[best_idx])
