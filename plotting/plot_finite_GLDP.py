# A numerical result for finite data space

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from utils.div import kl_f, tv_f, hellinger_f

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    'font.family': 'serif',
    'figure.constrained_layout.use': True,
    'axes.formatter.useoffset': False,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    # 'text.usetex': True,  # Disable LaTeX for Colab
})

def max_pmf_unifMollifier(k, eps):
    return np.minimum(np.exp(eps/2)/k, 1-(k-1)/k*np.exp(-eps/2))


def worst_fDiv_ours(k, eps, f):
    max_pmf = np.exp(eps)/(np.exp(eps)+k-1)
    return (1-max_pmf)*f(0) + max_pmf*f(1/max_pmf)

def worst_fDiv_local(k, eps, f):
    gamma = k / 2 - 1
    # print(f"gamma is {gamma}")

    r_1 = (np.exp(eps) + gamma) / (gamma * (gamma + 1))
    r_2 = (gamma * (np.exp(eps) + gamma)) / (np.exp(eps) * (gamma + 1))

    term1 = ((1 - r_1) / (r_2 - r_1)) * f(r_2)
    term2 = ((r_2 - 1) / (r_2 - r_1)) * f(r_1)

    return term1 + term2

def worst_fDiv_prev(k, eps, f):
    max_pmf = max_pmf_unifMollifier(k, eps)
    return (1-max_pmf)*f(0) + max_pmf*f(1/max_pmf)

def worst_fDiv_ours_GDP(k, nu, f):
    c2 = k
    c1 = 0
    

    def objective(beta):
        expb = np.exp(beta)
        phi1 = norm.cdf(-nu / 2 - beta / nu)
        phi2 = norm.cdf(-nu / 2 + beta / nu)
        numerator = expb + ((c2 - c1)/(1 - c1)) * (1 - expb * phi1 - phi2) - 1
        denominator = (1 - c1) * expb + c2 - 1
        return numerator / denominator

    # Minimize the objective to find the optimal beta (lambda^*)
    res = minimize_scalar(objective, bounds=(0, 20), method='bounded')
    

    beta_star = res.x
    lam_star = objective(beta_star) 

    # Compute r1 and r2
    r1 = c1 / (1 - (1 - c1) * lam_star)
    r2 = c2 / ((c2 - 1) * lam_star + 1)

    # Compute the convex combination
    w1 = (1 - r1) / (r2 - r1)
    w2 = (r2 - 1) / (r2 - r1)
    return w1 * f(r2) + w2 * f(r1)


def worst_fDiv_local_GDP(k, nu, f):
    
    gamma = k/2 - 1
    c2 = gamma
    c1 = 1/gamma
    

    def objective(beta):
        expb = np.exp(beta)
        phi1 = norm.cdf(-nu / 2 - beta / nu)
        phi2 = norm.cdf(-nu / 2 + beta / nu)
        numerator = expb + ((c2 - c1)/(1 - c1)) * (1 - expb * phi1 - phi2) - 1
        denominator = (1 - c1) * expb + c2 - 1
        return numerator / denominator

    # Minimize the objective to find the optimal beta (lambda^*)
    res = minimize_scalar(objective, bounds=(0, 20), method='bounded')
    
    beta_star = res.x
    lam_star = objective(beta_star) 

    # Compute r1 and r2
    r1 = c1 / (1 - (1 - c1) * lam_star)
    r2 = c2 / ((c2 - 1) * lam_star + 1)

    # Compute the convex combination
    w1 = (1 - r1) / (r2 - r1)
    w2 = (r2 - 1) / (r2 - r1)
    return w1 * f(r2) + w2 * f(r1)


def main(k: int, figname: str | None):
    if figname is None:
        figname = f'result_finite_k{k}.pdf'  # save as PDF

    nu_list = np.array([0.1, 0.5, 1, 2])
    f_list = [kl_f, tv_f, hellinger_f]
    f_name_list = ["KL", "TV", "Sq. Hel"]

    fig = plt.figure()

    for i in range(1, 4):
        plt.subplot(1, 3, i)
        f = f_list[i - 1]
        plt.title(f_name_list[i - 1])

        ours = np.array([worst_fDiv_ours_GDP(k, nu, f) for nu in nu_list])

        local = np.array([worst_fDiv_local_GDP(k, nu, f) for nu in nu_list])

        plot_data = pd.DataFrame({
            "Privacy budget $\\nu$": np.concatenate([nu_list, nu_list]),
            "Worst $f$-divergence": np.concatenate([ours, local]),
            "mech": ["Global"] * len(nu_list) + ["Local"] * len(nu_list)
        })

        plot = sns.barplot(data=plot_data, x="Privacy budget $\\nu$", y="Worst $f$-divergence", hue="mech",palette="Set1")
        plot.get_legend().remove()

        if i != 2:
            plot.set(xlabel=None)
        if i != 1:
            plot.set(ylabel=None)

    handles, labels = plot.get_legend_handles_labels()
    fig.legend(loc='lower center', ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.25, 0.005))
    fig.set_figheight(2.5)
    plt.tight_layout()

    plt.savefig(figname)  # Saves as PDF
    plt.show()   


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--figname", type=str, default=None)
    args = parser.parse_args()

    main(args.k, args.figname)