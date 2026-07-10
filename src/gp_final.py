"""
The coupled digital twin of the beamline (notebook 07).

`gp.py` and `acquisition.py` stay untouched. This module joins the two
previous emulators into ONE object with two heads over the same input
space (the unit cube of the 8 electrode voltages):

- transmission head (`gp_hits`): GaussianProcess on log1p(hits). It models
  the goal quantity, but alone it is blind on the ~90% of the space where
  hits = 0 (every zero looks the same).
- distance head (`gp_dist`): DistanceGP on log1p(mean distance to the
  detector center). Continuous everywhere, it orders precisely the plateau
  where the transmission head has no signal.

The coupling: the distance head becomes P(distance < threshold) -- the
probability that the beam lands near the detector -- and every decision
made on the transmission head (acquisition, control tasks) is gated by
that probability. The threshold self-tunes: it is the best quartile of
the observed distances, floored at close_mm, so early on the gate asks
"closer than the best quarter of what we've seen" instead of collapsing
to zero everywhere, and it tightens toward close_mm by itself as better
points arrive. With no distance data at all the gate opens fully
(p_close = 1) and the twin degrades gracefully to the notebook-05
behavior.
"""

import numpy as np
from scipy.stats import norm

from gp import GaussianProcess
from gp_distance import DistanceGP


class BeamlineEmulator:
    """One twin, two coupled heads. All inputs `U` live in the unit cube.

    Parameters
    ----------
    n_ions : int
        Ions fired per fly; upper admissible bound for the transmission.
    max_distance_mm : float
        Penalty/cap of the distance target (see DistanceGP).
    close_mm : float
        The floor of the gate's threshold: what "close to the detector"
        should eventually mean. The detector box is ~12 x 13 x 4 mm, so
        ~20 mm means the beam is essentially on target. Until the data
        actually gets that close, the gate uses the best quartile of the
        observed distances instead (see `p_close`).
    """

    def __init__(self, n_ions=500, max_distance_mm=500.0, close_mm=20.0):
        self.n_ions = int(n_ions)
        self.max_distance_mm = float(max_distance_mm)
        self.close_mm = float(close_mm)
        self.gp_hits = None
        self.gp_dist = None
        self._gate_threshold_mm = None

    @property
    def log_hits_bounds(self):
        """Admissible (low, high) for transmission predictions, in log1p space."""
        return (0.0, np.log1p(self.n_ions))

    def _to_log_hits(self, hits):
        h = np.asarray(hits, dtype=float).ravel()
        return np.log1p(np.clip(h, 0.0, self.n_ions))

    # ----- fitting -----------------------------------------------------------

    # Noise floor for both heads' hyperparameter fits, in the log1p target
    # space. Without it, the marginal likelihood can drive noise_var to ~0
    # and the GP becomes an exact interpolator that hallucinates huge means
    # (with understated stds) between training points -- every gated score
    # downstream then chases those ghosts. 1e-2 is far below the noise the
    # real data actually shows (~0.37 in notebook 05), so it only binds in
    # the pathological regime.
    _LOG_NOISE_VAR_BOUNDS = (np.log(1e-2), 3.0)

    def fit_transmission(self, U, hits, n_restarts=8, seed=0):
        """Fit the transmission head, ARD hyperparameters included."""
        self.gp_hits = GaussianProcess()
        self.gp_hits.fit_hyperparameters(
            U, self._to_log_hits(hits), n_restarts=n_restarts, seed=seed,
            log_noise_var_bounds=self._LOG_NOISE_VAR_BOUNDS)
        return self

    def fit_distance(self, U, distance_mm, n_restarts=8, seed=0):
        """Fit the distance head, ARD hyperparameters included."""
        self.gp_dist = DistanceGP(max_distance_mm=self.max_distance_mm)
        self.gp_dist.fit_distance_hyperparameters(
            U, distance_mm, n_restarts=n_restarts, seed=seed,
            log_noise_var_bounds=self._LOG_NOISE_VAR_BOUNDS)
        self._record_gate_threshold(distance_mm)
        return self

    def refit_transmission(self, U, hits):
        """Refit keeping the current hyperparameters (cheap, for active loops)."""
        gp = self.gp_hits
        self.gp_hits = GaussianProcess(lengthscale=gp.lengthscale,
                                       signal_var=gp.signal_var,
                                       noise_var=gp.noise_var)
        self.gp_hits.fit(U, self._to_log_hits(hits))
        return self

    def refit_distance(self, U, distance_mm):
        """Refit the distance head keeping the current hyperparameters."""
        gp = self.gp_dist
        self.gp_dist = DistanceGP(lengthscale=gp.lengthscale,
                                  signal_var=gp.signal_var,
                                  noise_var=gp.noise_var,
                                  max_distance_mm=self.max_distance_mm)
        self.gp_dist.fit_distance(U, distance_mm)
        self._record_gate_threshold(distance_mm)
        return self

    def _record_gate_threshold(self, distance_mm):
        d = np.clip(np.asarray(distance_mm, dtype=float), 0.0,
                    self.max_distance_mm)
        self._gate_threshold_mm = float(max(self.close_mm,
                                            np.quantile(d, 0.25)))

    # ----- prediction --------------------------------------------------------

    def predict_transmission(self, U):
        """(hits_mean, log_mean, log_std): expected ions on the detector,
        admissible in [0, n_ions]; the log-space pair feeds acquisitions."""
        log_mean, log_std = self.gp_hits.predict(U, bounds=self.log_hits_bounds)
        return np.expm1(log_mean), log_mean, log_std

    def predict_distance(self, U):
        """(mean_mm, lo_mm, hi_mm), or None while the distance head is unfit."""
        if self.gp_dist is None:
            return None
        return self.gp_dist.predict_distance(U)

    def p_close(self, U, close_mm=None):
        """P(distance < threshold) under the distance head's posterior.

        This is the coupling gate. The threshold is the best quartile of
        the observed distances, floored at close_mm: gating on the fixed
        floor while nothing has landed that close yet would score ~0
        everywhere and turn every argmax downstream into an arbitrary
        pick, and gating on the single best observation is nearly as harsh
        (P(beat the record) ~ 0 by definition). The quartile keeps the
        ranking informative and tightens toward close_mm by itself as
        better points arrive. Returns 1.0 everywhere while the distance
        head is unfit, so an uncoupled twin still works.
        """
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if self.gp_dist is None:
            return np.ones(len(U))
        if close_mm is None:
            close = (self.close_mm if self._gate_threshold_mm is None
                     else self._gate_threshold_mm)
        else:
            close = float(close_mm)
        mean_log, std_log = self.gp_dist.posterior(U)
        return norm.cdf((np.log1p(close) - mean_log) / std_log)

    def coupled_transmission(self, U):
        """Predicted transmission gated by proximity: expm1(mean) * P(close).

        The single scalar the control tasks optimize: high only where the
        transmission head expects hits AND the distance head believes the
        beam actually lands near the detector.
        """
        hits_mean, _, _ = self.predict_transmission(U)
        return hits_mean * self.p_close(U)

    def gradient_transmission(self, x):
        """Gradient of the transmission head's posterior mean (log space)."""
        return self.gp_hits.gradient(x)
