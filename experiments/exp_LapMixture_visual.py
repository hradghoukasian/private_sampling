import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from time import time

from utils.measure import ContinuousMeasure, LaplaceMixture
from utils.f_epsOptMech import OptMech_ConvexMixing
from utils.f_epsOptMech import Local_OptMech_ConvexMixing
from utils.gdp_mech import GDPOptMech
from utils.gdp_mech import Local_GDPOptMech
from utils.propMech import ProposedMech_Continuous, Local_ProposedMech_Continuous

# Matplotlib settings (clean and readable)
matplotlib.rcParams.update({
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 2)
})

if __name__ == "__main__":
    eps = 1
    abs_range = 2
    gaussian_var = 0.1
    mu = 1.5

    seed = 0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    laplace_centers = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = 2

    # Define original (target) distribution
    # target = GaussianRing(0, 1, gaussian_var, 3)
    target = LaplaceMixture(laplace_centers, b)

    # Define upper bound measure
    # def density_ub(x):
    #     max_val = 1 / (2 * np.pi * gaussian_var)
    #     return np.where(np.linalg.norm(x, axis=-1) < 1,
    #                     max_val,
    #                     max_val * np.exp(-(np.linalg.norm(x, axis=-1) - 1)**2 / (2 * np.sqrt(gaussian_var))))

    # Upper bound reference measure: Laplace(0, b) in 2D
    def density_ub(x):
        x = np.asarray(x)
        abs_sum = np.abs(x[..., 0]) + np.abs(x[..., 1])  # ||x||_1
        return (1 / (4 * b**2)) * np.exp(-abs_sum / b)

    def local_density_ub(x):

        k = 3

        x = np.asarray(x)
        norm_shape = x.shape[:-1]  # e.g. (500, 500) if input is (500, 500, 2)
        x_flat = x.reshape(-1, 2)  # reshape to (N, 2)

        # Generate k means on the unit circle
        angles = 2 * np.pi * np.arange(k) / k
        mus = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # shape (k, 2)

        # Compute squared distances: shape (N, k)
        dists_squared = np.sum((x_flat[:, None, :] - mus[None, :, :])**2, axis=-1)

        # Take min over i=1..k, then compute φ
        min_sq_dist = np.min(dists_squared, axis=-1)
        h0_vals = (1 / (2 * np.pi * gaussian_var)) * np.exp(-min_sq_dist / (2 * gaussian_var))

        return h0_vals.reshape(norm_shape)

    local_meas_ub = ContinuousMeasure(2, density_ub, [[-np.inf, np.inf], [-np.inf, np.inf]])
    meas_ub = ContinuousMeasure(2, density_ub, [[-np.inf, np.inf], [-np.inf, np.inf]])

    ############# Global Minimax
    # alpha = 2
    ## Global Proposed Mechanism
    t = time()
    global_propMech = ProposedMech_Continuous(eps, meas_ub, (1/3) *  np.exp(-1 / b) , 3 * np.exp(1 / b) )
    time_prep_prop = time() - t

    t = time()
    Q_global_prop = global_propMech(target)
    time_perturb_prop = time() - t

    # Local Proposed Mechanism
    t = time()
    local_propMech = Local_ProposedMech_Continuous(eps, local_meas_ub, np.exp(-1 / b), np.exp(1 / b))
    local_time_prep_prop = time() - t

    t = time()
    Q_local_prop = local_propMech(target)
    local_time_perturb_prop = time() - t


    # # Prepare the f_eps mechanism
    # t = time()
    # optMech = OptMech_ConvexMixing(eps, meas_ub, 0 , 1)
    # time_prep_feps = time() - t

    # # Run the mechanism on the target distribution
    # t = time()
    # Q_optMech = optMech(target)
    # time_perturb_feps = time() - t

    # Prepare the GDP mechanism
    t = time()
    gdpMech = GDPOptMech(mu, meas_ub, (1/3) *  np.exp(-1 / b) , 3 * np.exp(1 / b) )
    time_prep_gdp = time() - t

    # Run the mechanism on the target distribution
    t = time()
    Q_gdpMech = gdpMech(target)
    time_perturb_gdp = time() - t

    # ############# Local Minimax
    # # Prepare the f_eps mechanism
    # t = time()
    # local_optMech = Local_OptMech_ConvexMixing(eps, local_meas_ub, 1/3, 1)
    # local_time_prep_feps = time() - t

    # # Run the mechanism on the target distribution
    # t = time()
    # local_Q_optMech = local_optMech(target)
    # local_time_perturb_feps = time() - t

    # Prepare the GDP mechanism
    t = time()
    local_gdpMech = Local_GDPOptMech(mu, local_meas_ub, np.exp(-1 / b), np.exp(1 / b))
    local_time_prep_gdp = time() - t

    # Run the mechanism on the target distribution
    t = time()
    local_Q_gdpMech = local_gdpMech(target)
    local_time_perturb_gdp = time() - t


    # Report timing
    # print("Running time:")
    # print(f"Global Preparation (f_eps): {time_prep_feps:.2f} seconds")
    # print(f"Global Perturbation (f_eps): {time_perturb_feps:.2f} seconds")
    # print(f"Global Preparation (GDP): {time_prep_gdp:.2f} seconds")
    # print(f"Global Perturbation (GDP): {time_perturb_gdp:.2f} seconds")
    # print(f"Local Preparation (f_eps): {local_time_prep_feps:.2f} seconds")
    # print(f"Local Perturbation (f_eps): {local_time_perturb_feps:.2f} seconds")
    # print(f"Local Preparation (GDP): {local_time_prep_gdp:.2f} seconds")
    # print(f"Local Perturbation (GDP): {local_time_perturb_gdp:.2f} seconds")
    # print(f"Global Preparation (Proposed): {time_prep_prop:.2f} seconds")
    # print(f"Global Perturbation (Proposed): {time_perturb_prop:.2f} seconds")
    # print(f"Local Preparation (Proposed): {local_time_prep_prop:.2f} seconds")
    # print(f"Local Perturbation (Proposed): {local_time_perturb_prop:.2f} seconds")

    # Create a grid for visualization
    x = np.linspace(-abs_range, abs_range, 500)
    y = np.linspace(-abs_range, abs_range, 500)
    xx, yy = np.meshgrid(x, y)
    points = np.dstack((xx, yy))

    # Compute densities
    original_densities = target.density(points)
    # optMech_output = Q_optMech.density(points)
    gdpMech_output = Q_gdpMech.density(points)
    #local_optMech_output = local_Q_optMech.density(points)
    local_gdpMech_output = local_Q_gdpMech.density(points)
    global_prop_output = Q_global_prop.density(points)
    local_prop_output = Q_local_prop.density(points)


    # Save computed density grids for later use
    np.savez("LaplaceMixture_densities.npz",
        original=original_densities,
        global_pure=global_prop_output,
        local_pure=local_prop_output,
        global_gdp=gdpMech_output,
        local_gdp=local_gdpMech_output)


    fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows, 3 columns

    xticks = [-2, -1, 0, 1, 2]
    yticks = [-2, -1, 0, 1, 2]

    # First row: Original, Global Pure, Local Pure
    axes[0, 0].imshow(original_densities, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(global_prop_output, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Global Pure')

    axes[0, 2].imshow(local_prop_output, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Local Pure')

    # Second row: Original, Global GDP, Local GDP
    axes[1, 0].imshow(original_densities, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Original')

    axes[1, 1].imshow(gdpMech_output, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Global GDP')

    axes[1, 2].imshow(local_gdpMech_output, extent=(-abs_range, abs_range, -abs_range, abs_range),
                      cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Local GDP')

    # Set consistent ticks
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xticks(xticks)
            axes[i, j].set_yticks(yticks if j == 0 else [])  # Hide y-ticks except left column

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("LaplaceMixture_2x3Comparison.pdf", bbox_inches='tight')
    plt.savefig("LaplaceMixture_2x3Comparison.eps", bbox_inches='tight')
    plt.show()