import numpy as np
import torch
from tqdm import trange
from argparse import ArgumentParser

from utils.measure import ContinuousMeasure, LaplaceMixtureBounded
from utils.div_HighDim import kl, tv, hellinger
# Replace the following two imports with your own mechanism code
from utils.f_epsOptMech import OptMech_ConvexMixing, Local_OptMech_ConvexMixing


def sample_uniform_l1_ball(rng, dim, n_samples):
    """Sample n_samples points uniformly from the unit L1 ball in R^dim."""
    signs = rng.choice([-1, 1], size=(n_samples, dim))
    exp_samples = rng.exponential(scale=1.0, size=(n_samples, dim))
    normed = exp_samples / exp_samples.sum(axis=1, keepdims=True)
    points = signs * normed
    radii = rng.uniform(0, 1, size=(n_samples, 1))
    return points * radii  # Scale to L1 ball


# ---------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------
def run_experiment(
    eps="1.0",
    dim=2,
    mean_num_modes=3,
    max_num_modes=10,
    maxabs_mode=1,
    scale=2.0,
    size=100,
    seed=0,
):
    eps = float(eps)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # upper‑bounding Laplace density (same normalising constant!)
    def global_density_ub(x):
        return ((1.0 / (2.0 * scale)) ** dim) * np.exp(
            -np.linalg.norm(x, ord=1, axis=-1) / scale
        )

    box = [[-10.0, 10.0]] * dim
    meas_ub = ContinuousMeasure(dim, global_density_ub, box)

    # Instantiate mechanisms (user‑provided implementations)
    glob_loc_ratio = 3.0
    globalMech = OptMech_ConvexMixing(
        eps, meas_ub, (1 / glob_loc_ratio) * np.exp(-1 / scale), glob_loc_ratio * np.exp(1 / scale)
    )
    localMech = Local_OptMech_ConvexMixing(
        eps, meas_ub, np.exp(-1 / scale), np.exp(1 / scale)
    )

    # Storage
    results = np.zeros((size, 6))  # [KL_g, KL_l, TV_g, TV_l, Hel_g, Hel_l]

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    for i in trange(size, desc="experiments"):
        # random Laplace mixture
        num_modes = min(rng.poisson(mean_num_modes - 1) + 1, max_num_modes)
        centers = sample_uniform_l1_ball(rng, dim, num_modes)
        weights = rng.dirichlet(np.ones(num_modes))
        target = LaplaceMixtureBounded(centers, scale, weights, ranges=box).normalize_montecarlo()

        # Apply mechanisms + re‑normalise
        out_global = globalMech(target).normalize_montecarlo()
        out_local  = localMech(target).normalize_montecarlo()

        # Divergences
        results[i, 0] = kl(target, out_global)
        results[i, 1] = kl(target, out_local)
        results[i, 2] = tv(target, out_global)
        results[i, 3] = tv(target, out_local)
        results[i, 4] = hellinger(target, out_global)
        results[i, 5] = hellinger(target, out_local)

    # save
    filename = f"data_{dim}DLaplaceMix_feps_eps{eps}.npy"
    np.save(filename, results)
    print("Saved →", filename)


# ---------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eps", type=str, default="1.0")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--scale", type=float, default=2.0)

    run_experiment(**vars(parser.parse_args()))
