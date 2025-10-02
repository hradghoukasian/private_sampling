from __future__ import annotations
import numpy as np
import scipy.integrate as integrate


def _truncation(f, ranges):
    ranges_np = np.array(ranges)
    lb, ub = ranges_np[:, 0], ranges_np[:, 1]

    def _new_density(x):
        inside = np.logical_and(lb <= x, x <= ub).all(axis=-1)
        return np.where(inside, f(x), 0.0)

    return _new_density


# ---------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------
class ContinuousMeasure:
    """
    A minimal wrapper around an unnormalised density on R^d.
    """

    def __init__(self, dim, density, ranges, auto_truncate=False):
        self.dim = int(dim)
        self.ranges = np.array(ranges, dtype=float)
        self.density = _truncation(density, ranges) if auto_truncate else density

  
    def _density_packed(self, *x):
        return self.density(np.array(x))

    def total_mass(self):
        return integrate.nquad(self._density_packed, self.ranges)[0]

    def integrate(self, f):
        return integrate.nquad(
            lambda *x: f(np.array(x)) * self.density(np.array(x)), self.ranges
        )[0]

    
    def normalize(self):
        Z = self.total_mass()
        return ContinuousMeasure(
            self.dim, lambda x: self.density(x) / Z, self.ranges
        )

    def normalize_montecarlo(self, num_samples=10_000, rng=np.random.default_rng()):
        from .div_HighDim import monte_carlo_integrate  # lazy import
        Z = monte_carlo_integrate(lambda x: self.density(x), self.ranges, num_samples, rng)
        return ContinuousMeasure(
            self.dim, lambda x: self.density(x) / Z, self.ranges
        )

    def truncate(self, ranges):
        return ContinuousMeasure(self.dim, _truncation(self.density, ranges), ranges)



class GaussianMixture_sameVar(ContinuousMeasure):
    """
    Mixture of isotropic Gaussians with equal variance.
    """

    def __init__(self, peaks, var, weights):
        peaks = np.asarray(peaks, dtype=float)
        weights = np.asarray(weights, dtype=float)
        assert np.isclose(weights.sum(), 1.0)
        self.dim = peaks.shape[1]
        self.peaks = peaks
        self.var = float(var)
        self.weights = weights

        super().__init__(self.dim, self._density, [[-np.inf, np.inf]] * self.dim)

    # correct normalising constant
    def _density(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, self.dim)

        peaks_exp = self.peaks[np.newaxis, ...]  # (..., n_modes, dim)
        diff = x[..., np.newaxis, :] - peaks_exp
        exponent = -np.linalg.norm(diff, axis=-1) ** 2 / (2.0 * self.var)
        const = (2.0 * np.pi * self.var) ** (self.dim / 2.0)

        return np.sum(self.weights * np.exp(exponent), axis=-1) / const


# ---------------------------------------------------------------------
# Laplace mixtures (ℓ1‑norm exponent)
# ---------------------------------------------------------------------
class LaplaceMixture(ContinuousMeasure):
    """
    iid‑Laplace mixture with common scale `b` (ℓ1 norm).
    """

    def __init__(self, centers, b, weights=None):
        centers = np.asarray(centers, dtype=float)
        n_modes, dim = centers.shape

        if weights is None:
            weights = np.full(n_modes, 1.0 / n_modes)
        weights = np.asarray(weights, dtype=float)
        assert np.isclose(weights.sum(), 1.0)

        self.dim = dim
        self.centers = centers
        self.b = float(b)
        self.weights = weights

        super().__init__(self.dim, self._density, [[-np.inf, np.inf]] * dim)

    # correct normalising constant
    def _density(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, self.dim)

        x_exp = x[..., np.newaxis, :]              # (..., n_points, 1, dim)
        centers_exp = self.centers[np.newaxis, ...]  # (1, n_modes, dim)
        l1 = np.sum(np.abs(x_exp - centers_exp), axis=-1)  # (..., n_points, n_modes)

        const = (1.0 / (2.0 * self.b)) ** self.dim
        return np.sum(self.weights * const * np.exp(-l1 / self.b), axis=-1)


class LaplaceMixtureBounded(LaplaceMixture):
    """
    Same as LaplaceMixture but restricted to a finite box `ranges`.
    """

    def __init__(self, centers, b, weights=None, ranges=None):
        if ranges is None:
            ranges = [[-np.inf, np.inf]] * centers.shape[1]
        self._box_ranges = np.asarray(ranges, dtype=float)
        super().__init__(centers, b, weights)
        # overwrite with truncated density
        self.ranges = self._box_ranges
        self.density = _truncation(self.density, self.ranges)
