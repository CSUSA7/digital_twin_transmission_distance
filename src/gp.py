"""
A Gaussian process you build yourself.

This file is the core exercise of part 2. Two methods are left for you to write:
the kernel and the posterior. The notebook for part 2 walks you through both,
with the formulas and the array shapes you need. Everything else is here so you
can test your work as you go.

When you finish a method, save this file. If your notebook has autoreload on
(the setup cell turns it on), the change takes effect on the next cell you run.
"""

import numpy as np


class GaussianProcess:
    """A Gaussian process with a squared-exponential (RBF) kernel.

    Parameters
    ----------
    lengthscale : float or array of shape (d,)
        How far apart two inputs can be before the model treats them as
        unrelated. Small means wiggly, large means smooth. Pass a vector with
        one value per input dimension (ARD) to let each dimension have its
        own scale -- useful when the dimensions live on different physical
        ranges, e.g. some voltages in +-500V and others in +-1000V.
    signal_var : float
        The variance of the function, that is, how far it swings from its mean.
    noise_var : float
        The variance of the observation noise. Keep a tiny value even for
        noise-free data, so the matrix stays invertible.
    """

    def __init__(self, lengthscale=0.2, signal_var=1.0, noise_var=1e-6):
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X = None
        self.y = None
        self.mean_y = 0.0

    @staticmethod
    def _as_2d(X):
        """Treat a 1-D array as a column of n one-dimensional points."""
        X = np.asarray(X, dtype=float)
        return X[:, None] if X.ndim == 1 else X

    def kernel(self, A, B):
        """Covariance between every row of A and every row of B.

        A has shape (n, d), B has shape (m, d). The result has shape (n, m).

        The squared-exponential kernel is
            k(a, b) = signal_var^2 * exp(-0.5 * ||(a - b) / lengthscale||^2).

        `lengthscale` may be a scalar (isotropic) or a length-d vector (ARD,
        one scale per dimension). Dividing by it before computing distances
        makes both cases fall out of the same broadcasting.

        A clean way to get all the pairwise squared distances without a loop:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b
        Build that as an (n, m) array, clip any tiny negative values to zero,
        then apply the exponential.
        """
        A = self._as_2d(A) / self.lengthscale
        B = self._as_2d(B) / self.lengthscale
        sq_norm_A = np.sum(A**2, axis=1)[:, None]
        sq_norm_B = np.sum(B**2, axis=1)[None, :]
        sq_dists = sq_norm_A + sq_norm_B - 2 * A @ B.T
        sq_dists = np.clip(sq_dists, 0, None)
        return self.signal_var**2 * np.exp(-0.5 * sq_dists)


    def fit(self, X, y, center=True):
        """Store the data and factor the training covariance once.

        The GP itself has a zero prior mean. When the target is not
        naturally centered at zero (e.g. beam spread, not a sparse hit
        count), set `center=True` (the default) so the model fits the
        residual `y - mean(y)` and reports predictions relative to that
        mean. Pass `center=False` to keep the old zero-mean behavior.

        This is given. It calls your kernel, so it starts working as soon as
        your kernel is correct. We use a Cholesky factor instead of a raw
        inverse because it is faster and far more stable numerically.
        """
        self.X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        self.mean_y = float(y.mean()) if center else 0.0
        self.y = y - self.mean_y
        K = self.kernel(self.X, self.X) + self.noise_var * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def posterior(self, X_test):
        """Posterior mean and standard deviation at the test points.

        X_test has shape (m, d). Return (mean, std), each of length m.

        With K already factored in fit() as self.L (lower triangular) and
        self.alpha = K^{-1} y, the posterior at test points X* is

            K_s   = kernel(X_train, X_test)          shape (n, m)
            mean  = K_s^T . alpha                     length m
            v     = solve(L, K_s)                     shape (n, m)
            var   = signal_var - sum(v * v, axis=0)   length m
            std   = sqrt(max(var, small_positive))

        The diagonal of kernel(X_test, X_test) equals signal_var for the RBF
        kernel, which is why the variance line is so short.
        """
        X_test = self._as_2d(X_test)
        K_s   = self.kernel(self.X, X_test)          # shape (n, m)
        mean  = K_s.T @ self.alpha + self.mean_y      # length m
        v     = np.linalg.solve(self.L, K_s)         # shape (n, m)
        var   = self.signal_var**2 - np.sum(v * v, axis=0) # length m the sigma squared is missing
        std   = np.sqrt(np.maximum(var, 1e-12))
        return mean, std

    def predict(self, X_test, bounds=None):
        """Posterior mean and standard deviation, clipped to physical bounds.

        `bounds`, if given, is a `(low, high)` pair (or arrays of those, one
        per output) used to enforce admissible predictions -- e.g. a hit
        count can never be negative or exceed the number of ions fired.
        `posterior()` stays the "raw" GP output used by sampling and the
        gradient; use `predict()` when reporting numbers or driving a
        control task.
        """
        mean, std = self.posterior(X_test)
        if bounds is not None:
            low, high = bounds
            mean = np.clip(mean, low, high)
        return mean, std

    # ----- given helpers -----------------------------------------------------

    def sample_prior(self, X_test, n_samples=3, rng=None):
        """Draw whole functions from the prior at the test points.

        Given. It only needs your kernel. Each row of the result is one sampled
        function evaluated at X_test.
        """
        rng = np.random.default_rng() if rng is None else rng
        X_test = self._as_2d(X_test)
        K = self.kernel(X_test, X_test) + 1e-9 * np.eye(len(X_test))
        L = np.linalg.cholesky(K)
        return (L @ rng.standard_normal((len(X_test), n_samples))).T

    def sample_posterior(self, X_test, n_samples=3, rng=None):
        """Draw whole functions from the posterior. Given; needs kernel and fit."""
        rng = np.random.default_rng() if rng is None else rng
        X_test = self._as_2d(X_test)
        K_s = self.kernel(self.X, X_test)
        mean = K_s.T @ self.alpha + self.mean_y
        v = np.linalg.solve(self.L, K_s)
        cov = self.kernel(X_test, X_test) - v.T @ v + 1e-9 * np.eye(len(X_test))
        L = np.linalg.cholesky(cov)
        return mean[None, :] + (L @ rng.standard_normal((len(X_test), n_samples))).T

    def log_marginal_likelihood(self):
        """Bonus. The evidence for the current hyperparameters.

        Useful for choosing the lengthscale by maximizing it. The formula is
            -0.5 * y^T alpha - sum(log(diag(L))) - 0.5 * n * log(2 pi).
        Delete the raise and return that once you want to try the bonus.
        """
        # TODO (bonus): return the log marginal likelihood.
        #raise NotImplementedError("Bonus: write the log marginal likelihood.")
        log_likelihood = -0.5 * self.y.T @ self.alpha
        log_likelihood -= np.sum(np.log(np.diag(self.L)))
        log_likelihood -= 0.5 * len(self.y) * np.log(2 * np.pi)
        return log_likelihood

    def fit_hyperparameters(self, X, y, n_restarts=5, seed=0, center=True,
                             log_lengthscale_bounds=(-5.0, 5.0),
                             log_signal_var_bounds=(-5.0, 5.0),
                             log_noise_var_bounds=(-12.0, 3.0)):
        """Choose (lengthscale per dimension, signal_var, noise_var) by
        maximizing the log marginal likelihood, then fit with the winner.

        Only feasible once the kernel supports ARD (a lengthscale per input
        dimension); with d dimensions this optimizes d + 2 numbers. All are
        optimized in log-space so they stay positive, with `n_restarts`
        random starts because the likelihood surface is not convex.

        The `log_*_bounds` keep each parameter inside a physically sane
        range (e.g. for inputs mapped to the unit cube, a lengthscale past
        ~150 is already "this dimension does not matter", not a value worth
        chasing to infinity). Without bounds, a dimension the data says
        nothing about can drive its lengthscale toward float overflow.
        """
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        d = X.shape[1]

        bounds = [log_lengthscale_bounds] * d + [log_signal_var_bounds, log_noise_var_bounds]

        def neg_lml(log_theta):
            theta = np.exp(log_theta)
            self.lengthscale = theta[:d]
            self.signal_var = theta[d]
            self.noise_var = theta[d + 1]
            try:
                self.fit(X, y, center=center)
            except np.linalg.LinAlgError:
                return 1e10
            return -self.log_marginal_likelihood()

        best = None
        for _ in range(n_restarts):
            log_theta0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            result = minimize(neg_lml, log_theta0, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result

        theta = np.exp(best.x)
        self.lengthscale = theta[:d]
        self.signal_var = theta[d]
        self.noise_var = theta[d + 1]
        return self.fit(X, y, center=center)

    def gradient(self, x):
        """Gradient of the posterior mean at a single point x, shape (d,).

        The RBF kernel is differentiable in closed form:
            d k(x, x_i) / dx = -k(x, x_i) * (x - x_i) / lengthscale^2
            d mean(x) / dx   = sum_i alpha_i * d k(x, x_i) / dx
        Useful to check the emulator against a SIMION finite difference, and
        to walk uphill/downhill on the emulator directly instead of only
        scoring random candidates.
        """
        x = np.asarray(x, dtype=float).ravel()
        diff = (x[None, :] - self.X) / (np.asarray(self.lengthscale) ** 2)  # (n, d)
        k = self.kernel(x[None, :], self.X).ravel()                        # (n,)
        dk_dx = -diff * k[:, None]                                         # (n, d)
        return dk_dx.T @ self.alpha
