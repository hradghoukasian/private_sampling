#Implementation of GLDP optimal mechanisms

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from .measure import ContinuousMeasure
from math import ceil

class Local_GDPOptMech:
    def __init__(self, mu, base_measure, density_lb, density_ub):
        if density_lb >= 1 or density_ub < 1:
            raise ValueError("Require density_lb < 1 < density_ub for the lambda formula to be valid.")

        base_measure_int = base_measure.total_mass()
        self.base_measure_norm = ContinuousMeasure(base_measure.dim, lambda x: base_measure.density(x) / base_measure_int, base_measure.ranges)
        lb_norm = density_lb * base_measure_int
        ub_norm = density_ub * base_measure_int

        # self.base_measure = base_measure
        self.mu = mu
        # self.c1 = lb_norm
        self.c2: int = max(ceil(ub_norm), ceil(1.0 / lb_norm))
        self.c1 = 1/ self.c2

        print(f"GDP Local c2 is: {self.c2}")
        print(f"GDP Local c1 is: {self.c1} \n")

        # Compute lambda_star_GDP via minimization
        self.lambda_ = self._compute_lambda_star()

        # if not (0 <= self.lambda_ <= 1):
        #     raise ValueError(f"GDP Local Computed lambda = {self.lambda_} is not in [0,1]")

    def _compute_lambda_star(self):
        mu = self.mu
        c1 = self.c1
        c2 = self.c2
        alpha = (c2 - c1) / (1 - c1)

        def objective(beta):
            expb = np.exp(beta)
            phi1 = norm.cdf(-mu / 2 - beta / mu)
            phi2 = norm.cdf(-mu / 2 + beta / mu)
            numerator = (
                expb + alpha * (1 - expb * phi1 - phi2) - 1
            )
            denominator = (1 - c1) * expb + c2 - 1
            return numerator / denominator

        result = minimize_scalar(objective, bounds=(0, 50), method='bounded')

        print(result,"\n")
        return result.fun

    def __call__(self, dist):
        return ContinuousMeasure(
            dim=dist.dim,
            ranges=dist.ranges,
            density=lambda x: self.lambda_ * dist.density(x) + (1 - self.lambda_) * self.base_measure_norm.density(x)
        )


class GDPOptMech:
    def __init__(self, mu, base_measure, density_lb, density_ub):
        if density_lb >= 1 or density_ub < 1:
            raise ValueError("Require density_lb < 1 < density_ub for the lambda formula to be valid.")

        base_measure_int = base_measure.total_mass()
        self.base_measure_norm = ContinuousMeasure(base_measure.dim, lambda x: base_measure.density(x) / base_measure_int, base_measure.ranges)
        lb_norm = density_lb * base_measure_int
        ub_norm = density_ub * base_measure_int

        # self.base_measure = base_measure
        self.mu = mu
        # self.c1 = lb_norm
        self.c2 = ub_norm
        self.c1 = lb_norm

        print(f"GDP Global c2 is: {self.c2}")
        print(f"GDP Global c1 is: {self.c1} \n")

        # Compute lambda_star_GDP via minimization
        self.lambda_ = self._compute_lambda_star()

        # if not (0 <= self.lambda_ <= 1):
        #     raise ValueError(f"Global GDP Computed lambda = {self.lambda_} is not in [0,1]")

    def _compute_lambda_star(self):
        mu = self.mu
        c1 = self.c1
        c2 = self.c2
        alpha = (c2 - c1) / (1 - c1)

        def objective(beta):
            expb = np.exp(beta)
            phi1 = norm.cdf(-mu / 2 - beta / mu)
            phi2 = norm.cdf(-mu / 2 + beta / mu)
            numerator = (
                expb + alpha * (1 - expb * phi1 - phi2) - 1
            )
            denominator = (1 - c1) * expb + c2 - 1
            return numerator / denominator

        result = minimize_scalar(objective, bounds=(0, 50), method='bounded')

        print(result,"\n")
        return result.fun

    def __call__(self, dist):
        return ContinuousMeasure(
            dim=dist.dim,
            ranges=dist.ranges,
            density=lambda x: self.lambda_ * dist.density(x) + (1 - self.lambda_) * self.base_measure_norm.density(x)
        )