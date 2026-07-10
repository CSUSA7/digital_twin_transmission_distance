"""
Acquisition and control tasks for the coupled twin (notebook 07).

`acquisition.py` stays untouched; the candidate generators are reused from
`acquisition_distance.py`. Every score here is the transmission-head score
GATED by the distance head's probability of landing near the detector
(`emulator.p_close`), so the twin stops spending budget on the hits = 0
plateau just because it *looks* unexplored.
"""

import numpy as np

from acquisition_distance import (expected_improvement_min, local_candidates,
                                  sobol_candidates)


def coupled_expected_improvement(emulator, U, best_log_hits, xi=0.01):
    """EI of the transmission head times P(close).

    The transmission head maximizes, and expected_improvement_min is written
    for minimization, so the mean and the incumbent enter negated (same trick
    as notebook 05). The gate then multiplies: a point can only score high if
    the hits model expects improvement AND the distance model believes the
    beam gets near the detector there.
    """
    mean, std = emulator.gp_hits.posterior(U)
    ei = expected_improvement_min(-mean, std, best=-float(best_log_hits), xi=xi)
    return ei * emulator.p_close(U)


def propose_next_point(emulator, U_observed, log_hits_observed, n_sobol=4096,
                       n_incumbents=5, n_local=256, radius=0.05, xi=0.01,
                       seed=None):
    """One coupled-EI step for the active-learning loop.

    Candidates = global scrambled-Sobol sweep + Gaussian clouds around the
    `n_incumbents` best observed points (highest hits). Returns
    (x_next, score); a score collapsing toward zero means the loop has
    converged (or the gate closed everywhere -- check p_close if in doubt).
    """
    rng = np.random.default_rng(seed)
    log_hits = np.ravel(log_hits_observed)
    incumbents = U_observed[np.argsort(log_hits)[::-1][:n_incumbents]]
    candidates = np.vstack([
        sobol_candidates(n_sobol, U_observed.shape[1], seed=rng),
        local_candidates(incumbents, n_per_center=n_local, radius=radius,
                         rng=rng),
    ])
    score = coupled_expected_improvement(emulator, candidates,
                                         best_log_hits=float(log_hits.max()),
                                         xi=xi)
    best_idx = int(np.argmax(score))
    return candidates[best_idx], float(score[best_idx])


def conservative_coupled_transmission(emulator, U, beta=1.0):
    """Gated lower-confidence transmission: expm1(mean - beta*std) * P(close).

    The plain posterior mean rewards extrapolation: far from the data the
    GP can hallucinate a high mean with nothing to contradict it (the
    notebook-05 direction task failed exactly there). Discounting by
    beta * log-space std makes a point score high only if the model is
    confident about it, and the proximity gate then vetoes regions where
    the beam does not even approach the detector.
    """
    _, log_mean, log_std = emulator.predict_transmission(U)
    lcb = np.clip(log_mean - beta * log_std, 0.0, None)
    return np.expm1(lcb) * emulator.p_close(U)


def best_predicted_point(emulator, n_candidates=8192, seed=0,
                         refine_steps=200, lr=0.05, beta=1.0):
    """Direction task: argmax of the gated, uncertainty-discounted transmission.

    The training points themselves compete as candidates, so the proposal
    can never score worse than "repeat the best known point"; the Sobol
    sweep then looks for something better, and the refinement walks up the
    transmission head's analytic gradient but only keeps steps that improve
    that same score -- so it can neither climb out of the region the
    distance head trusts nor into one the hits head merely guesses about.
    """
    d = emulator.gp_hits.X.shape[1]
    candidates = np.vstack([
        sobol_candidates(n_candidates, d, seed=seed),
        emulator.gp_hits.X,
    ])
    score = conservative_coupled_transmission(emulator, candidates, beta=beta)
    x = candidates[int(np.argmax(score))].copy()
    best_x, best_score = x.copy(), float(np.max(score))
    for _ in range(refine_steps):
        x = np.clip(x + lr * emulator.gradient_transmission(x), 0.0, 1.0)
        s = float(conservative_coupled_transmission(
            emulator, x.reshape(1, -1), beta=beta)[0])
        if s > best_score:
            best_x, best_score = x.copy(), s
    return best_x


def match_predicted_point(emulator, target_hits, beta=1.0, p_min=0.5,
                          n_candidates=8192, seed=0):
    """Inverse-setpoint task: predicted transmission closest to `target_hits`.

    Uncertainty is discounted (beta * log-space std) so the twin does not
    propose a point it does not trust, and only candidates with
    P(close) >= p_min compete. If nothing passes the gate (distance head
    still uncertain everywhere), the top decile of P(close) competes
    instead, so the task always returns something.
    """
    d = emulator.gp_hits.X.shape[1]
    candidates = sobol_candidates(n_candidates, d, seed=seed)
    hits_mean, _, log_std = emulator.predict_transmission(candidates)
    p = emulator.p_close(candidates)
    admissible = p >= p_min
    if not admissible.any():
        admissible = p >= np.quantile(p, 0.9)
    score = -(np.abs(hits_mean - float(target_hits)) + beta * log_std)
    score = np.where(admissible, score, -np.inf)
    return candidates[int(np.argmax(score))]
