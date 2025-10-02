# Definitions of f-divergences

from scipy.integrate import nquad
import numpy as np

def kl(p, q):
    epsilon = 1e-10  # Small epsilon to avoid log(0)
    def integrand(*x):
        p_val = p._density_packed(*x)
        q_val = q._density_packed(*x)
        p_val = np.maximum(p_val, epsilon)
        q_val = np.maximum(q_val, epsilon)
        return p_val * (np.log(p_val) - np.log(q_val))
    return nquad(integrand, p.ranges)[0]

def tv(p, q):
    def integrand(*x):
        return np.abs(p._density_packed(*x) - q._density_packed(*x))
    return nquad(integrand, p.ranges)[0] / 2

def chi_square(p, q):
    def integrand(*x):
        q_val = np.maximum(q._density_packed(*x), 1e-10)
        return p._density_packed(*x) ** 2 / q_val
    return nquad(integrand, p.ranges)[0] - 1

def hellinger(p, q):
    def integrand(*x):
        return (np.sqrt(p._density_packed(*x)) - np.sqrt(q._density_packed(*x)))**2
    return nquad(integrand, p.ranges)[0] / 2

def kl_f(x):
    return np.where(np.isclose(x, 0), 0.0, x * np.log(x))

def tv_f(x):
    return np.abs(x - 1) / 2

def hellinger_f(x):
    return 1 - np.sqrt(x)
