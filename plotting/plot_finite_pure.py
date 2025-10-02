# A numerical result for finite data space

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
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



def main(k: int, figname: str | None):
    if figname is None:
        figname = f'result_finite_k{k}.pdf'  # save as PDF

    eps_list = np.array([0.1, 0.5, 1, 2])
    f_list = [kl_f, tv_f, hellinger_f]
    f_name_list = ["KL", "TV", "Sq. Hel"]

    fig = plt.figure()

    for i in range(1, 4):
        plt.subplot(1, 3, i)
        f = f_list[i - 1]
        plt.title(f_name_list[i - 1])

        ours = worst_fDiv_ours(k, eps_list, f)
        local = np.array([worst_fDiv_local(k, eps, f) for eps in eps_list])

        plot_data = pd.DataFrame({
            "Privacy budget $\\epsilon$": np.concatenate([eps_list, eps_list]),
            "Worst $f$-divergence": np.concatenate([ours, local]),
            "mech": ["Global"] * len(eps_list) + ["Local"] * len(eps_list)
        })

        plot = sns.barplot(data=plot_data, x="Privacy budget $\\epsilon$", y="Worst $f$-divergence", hue="mech",palette="Set1")
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