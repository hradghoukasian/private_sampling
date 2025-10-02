"""
High‑dimensional Monte‑Carlo estimators for common f‑divergences.
"""

import numpy as np

seed = 42
rng = np.random.default_rng(seed)
# ---------------------------------------------------------------------
# Basic Monte‑Carlo integrator over a hyper‑rectangle
# ---------------------------------------------------------------------
def monte_carlo_integrate(f, ranges, num_samples=10000, rng=np.random.default_rng()):
    dim   = len(ranges)
    lower = np.array([lo for lo, _ in ranges])
    upper = np.array([hi for _, hi in ranges])
    volume = np.prod(upper - lower)

    samples = rng.uniform(lower, upper, size=(num_samples, dim))
    return volume * np.mean(f(samples))


# ---------------------------------------------------------------------
# Pointwise f‑functions (for diagnostics, not used directly below)
# ---------------------------------------------------------------------
def kl_f(x):
    return np.where(np.isclose(x, 0), 0.0, x * np.log(x))


def tv_f(x):
    return np.abs(x - 1) / 2.0


def hellinger_f(x):
    return 1.0 - np.sqrt(x)


# ---------------------------------------------------------------------
# Monte‑Carlo f‑divergence estimators for ContinuousMeasure objects
# ---------------------------------------------------------------------
_EPS = 1e-10  # numerical safeguard


def kl(p, q, num_samples=100000, rng=rng):
    def _integrand(x):
        p_vals = p.density(x)
        q_vals = q.density(x)
        ratio = np.where(q_vals > 0, p_vals / q_vals, 1.0)  # 1.0 means 0 log 0 = 0
        log_ratio = np.where(p_vals > 0, np.log(ratio), 0.0)
        return np.where(p_vals > 0, p_vals * log_ratio, 0.0)

    return monte_carlo_integrate(_integrand, p.ranges, num_samples, rng)


def tv(p, q, num_samples=100000, rng=rng):
    def _integrand(x):
        return np.abs(p.density(x) - q.density(x))

    return 0.5 * monte_carlo_integrate(_integrand, p.ranges, num_samples, rng)


def hellinger(p, q, num_samples=100000, rng=rng):
    def _integrand(x):
        return (np.sqrt(p.density(x)) - np.sqrt(q.density(x))) ** 2

    return 0.5 * monte_carlo_integrate(_integrand, p.ranges, num_samples, rng)


def chi_square(p, q, num_samples=10000, rng=rng):
    def _integrand(x):
        p_vals = np.maximum(p.density(x), _EPS)
        q_vals = np.maximum(q.density(x), _EPS)
        return (p_vals ** 2) / q_vals

    return monte_carlo_integrate(_integrand, p.ranges, num_samples, rng) - 1.0
