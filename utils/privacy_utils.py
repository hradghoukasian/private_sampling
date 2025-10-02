import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def compute_delta(epsilon, mu):
    """
    Compute δ given ε and μ using the formula:
        δ(ε, μ) = Φ(-ε/μ + μ/2) - e^ε * Φ(-ε/μ - μ/2)

    Parameters:
        epsilon (float): The privacy parameter ε
        mu (float): The Gaussian DP parameter μ

    Returns:
        float: The corresponding δ value
    """
    term1 = norm.cdf((-epsilon / mu) + (mu / 2))
    term2 = np.exp(epsilon) * norm.cdf((-epsilon / mu) - (mu / 2))
    return term1 - term2

def compute_mu(epsilon, delta, mu_bounds=(1e-5, 100), tol=1e-10):
    """
    Compute μ given ε and δ by solving:
        δ = Φ(-ε/μ + μ/2) - e^ε * Φ(-ε/μ - μ/2)

    Parameters:
        epsilon (float): The privacy parameter ε
        delta (float): The target δ
        mu_bounds (tuple): Bracket interval to search for μ
        tol (float): Tolerance for the root solver

    Returns:
        float or None: The solution μ or None if solver fails
    """
    def f(mu):
        return compute_delta(epsilon, mu) - delta

    result = root_scalar(f, bracket=mu_bounds, method='bisect', xtol=tol)
    return result.root if result.converged else None

def compute_epsilon(mu, delta, eps_bounds=(1e-5, 50), tol=1e-10):
    """
    Compute ε given μ and δ by solving:
        δ = Φ(-ε/μ + μ/2) - e^ε * Φ(-ε/μ - μ/2)

    Parameters:
        mu (float): The Gaussian DP parameter μ
        delta (float): The target δ
        eps_bounds (tuple): Bracket interval to search for ε
        tol (float): Tolerance for the root solver

    Returns:
        float or None: The solution ε or None if solver fails
    """
    def f(eps):
        return compute_delta(eps, mu) - delta

    result = root_scalar(f, bracket=eps_bounds, method='bisect', xtol=tol)
    return result.root if result.converged else None