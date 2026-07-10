"""
Gaussian process for the continuous auxiliary objective of notebook 06:
the mean distance (in mm) from the recorded ion splats to the center of
the detector.

`gp.py` and `acquisition.py` stay untouched (they are the deliverables of
parts 2-3); this module builds on top of them. `DistanceGP` wraps the
generic `GaussianProcess` with the log1p warping and the physical bounds
of the distance target, so the notebook talks in millimeters while the GP
works in a well-behaved space.
"""

import numpy as np

from gp import GaussianProcess


class DistanceGP(GaussianProcess):
    """A GaussianProcess fit on log1p(distance_mm) instead of the raw target.

    Why the warping: the distance is positive, spans from fractions of a
    millimeter (beam focused on the detector) to hundreds (beam splatting
    on the first lens), and the region that matters most is near zero.
    log1p compresses the far field and expands the near field, which is
    exactly where we want the model to spend its resolution. `expm1`
    undoes it when reporting.

    Parameters are those of GaussianProcess plus:

    max_distance_mm : float
        Penalty value used when SIMION records no ion at all, and the
        upper admissible bound when reporting predictions in mm.
    """

    def __init__(self, lengthscale=0.3, signal_var=1.0, noise_var=1e-2,
                 max_distance_mm=500.0):
        super().__init__(lengthscale=lengthscale, signal_var=signal_var,
                         noise_var=noise_var)
        self.max_distance_mm = float(max_distance_mm)

    @property
    def log_bounds(self):
        """Admissible (low, high) for predictions, in log1p space."""
        return (0.0, np.log1p(self.max_distance_mm))

    def _to_log(self, distance_mm):
        d = np.asarray(distance_mm, dtype=float).ravel()
        return np.log1p(np.clip(d, 0.0, self.max_distance_mm))

    def fit_distance(self, U, distance_mm, center=True):
        """Fit on log1p of the clipped distance. `U` lives in the unit cube."""
        return self.fit(U, self._to_log(distance_mm), center=center)

    def fit_distance_hyperparameters(self, U, distance_mm, n_restarts=8,
                                     seed=0, center=True,
                                     log_noise_var_bounds=(-12.0, 3.0)):
        """Choose (ARD lengthscales, signal_var, noise_var) by maximizing
        the log marginal likelihood in log1p space, then fit with the winner.
        `log_noise_var_bounds` passes through to fit_hyperparameters, e.g.
        to impose a noise floor.
        """
        return self.fit_hyperparameters(U, self._to_log(distance_mm),
                                        n_restarts=n_restarts, seed=seed,
                                        center=center,
                                        log_noise_var_bounds=log_noise_var_bounds)

    def predict_distance(self, U):
        """Prediction in real mm units: (mean_mm, lo_mm, hi_mm).

        `lo_mm`/`hi_mm` are the 1-sigma band mapped through expm1 (exact,
        because the warping is monotone). `mean_mm` is the median of the
        log-normal-ish posterior rather than its expectation -- good
        enough for ranking and reporting, and it never leaves
        [0, max_distance_mm].
        """
        mean_log, std_log = self.predict(U, bounds=self.log_bounds)
        lo, hi = self.log_bounds
        mean_mm = np.expm1(mean_log)
        lo_mm = np.expm1(np.clip(mean_log - std_log, lo, hi))
        hi_mm = np.expm1(np.clip(mean_log + std_log, lo, hi))
        return mean_mm, lo_mm, hi_mm
