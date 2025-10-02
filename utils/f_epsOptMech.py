# Implementation of optimal mechanism for $g = g_eps$ 
 
import numpy as np
from .measure import ContinuousMeasure
from math import ceil

class Local_OptMech_ConvexMixing:
    def __init__(self, eps, base_measure, density_lb, density_ub):
        exp_eps = np.exp(eps)

        if density_lb >= 1 or density_ub < 1:
            raise ValueError("Require density_lb < 1 < density_ub for the lambda formula to be valid.")

        base_measure_int = base_measure.total_mass()
        self.base_measure_norm = ContinuousMeasure(base_measure.dim, lambda x: base_measure.density(x) / base_measure_int, base_measure.ranges)
        lb_norm = density_lb * base_measure_int
        ub_norm = density_ub * base_measure_int

        self.c2: int = max(ceil(ub_norm), ceil(1.0 / lb_norm))
        self.c1 = 1/ self.c2
        
        print(f"f_eps Local c2 is: {self.c2}")
        print(f"f_eps Local c1 is: {self.c1} \n")

        self.lambda_ = (exp_eps - 1) / ((1 - self.c1) * exp_eps + self.c2 - 1)
        #self.base_measure = base_measure
        print(f"lambda_f_eps is: {self.lambda_} \n")

        if not (0 <= self.lambda_ <= 1):
            raise ValueError(f"f_eps Local Computed lambda = {self.lambda_} is not in [0,1]")

    def __call__(self, dist):
        return ContinuousMeasure(
            dim=dist.dim,
            ranges=dist.ranges,
            density=lambda x: self.lambda_ * dist.density(x) + (1 - self.lambda_) * self.base_measure_norm.density(x)
        )


class OptMech_ConvexMixing:
    def __init__(self, eps, base_measure, density_lb, density_ub):
        exp_eps = np.exp(eps)

        if density_lb >= 1 or density_ub < 1:
            raise ValueError("Require density_lb < 1 < density_ub for the lambda formula to be valid.")

        base_measure_int = base_measure.total_mass()
        self.base_measure_norm = ContinuousMeasure(base_measure.dim, lambda x: base_measure.density(x) / base_measure_int, base_measure.ranges)
        lb_norm = density_lb * base_measure_int
        ub_norm = density_ub * base_measure_int

        self.c2 = ub_norm
        self.c1 = lb_norm
        
        print(f"f_eps Global c2 is: {self.c2}")
        print(f"f_eps Global c1 is: {self.c1} \n")

        self.lambda_ = (exp_eps - 1) / ((1 - self.c1) * exp_eps + self.c2 - 1)
        #self.base_measure = base_measure
        print(f"Global lambda_f_eps is: {self.lambda_} \n")

        if not (0 <= self.lambda_ <= 1):
            raise ValueError(f"Computed lambda = {self.lambda_} is not in [0,1]")

    def __call__(self, dist):
        return ContinuousMeasure(
            dim=dist.dim,
            ranges=dist.ranges,
            density=lambda x: self.lambda_ * dist.density(x) + (1 - self.lambda_) * self.base_measure_norm.density(x)
        )