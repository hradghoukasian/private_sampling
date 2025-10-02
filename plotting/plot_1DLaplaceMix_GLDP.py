# Plotting the results of the 1D Laplace mixture experiment (nu-based version)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from utils.div import kl_f, tv_f, hellinger_f

# Styling
matplotlib.rcParams.update({
    'font.family': 'serif',
    'figure.constrained_layout.use': True,
    'axes.formatter.useoffset': False,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    # 'text.usetex': True  # Enable this if you want full LaTeX rendering
})

def main(figname: str):
    nu_list = ['0.1','0.5','1.0','2.0']  # Adjust as needed
    f_list = [kl_f, tv_f, hellinger_f]
    f_name_list = ["KL", "TV", "Sq. Hel"]

    results = []

    for nu in nu_list:
        data = np.load(f'data_1DLaplaceMix_nu{nu}.npy')
        worst_fdiv = data.max(axis=0)
        results.append(pd.DataFrame({
            'Privacy parameter $\\nu$': [float(nu)] * 6,
            'Worst $f$-divergence': worst_fdiv,
            'mech': ['Global', 'Local'] * 3,
            'f': ['KL', 'KL', 'TV', 'TV', 'Sq. Hel', 'Sq. Hel']
        }))

    result = pd.concat(results)
    fig = plt.figure()

    for i in range(1, 4):
        plt.subplot(1, 3, i)
        plt.title(f_name_list[i - 1])

        plot_data = result[result['f'] == f_name_list[i - 1]]
        plot = sns.barplot(
            data=plot_data,
            x="Privacy parameter $\\nu$",
            y="Worst $f$-divergence",
            hue="mech",
            palette="Set1"
        )
        plot.get_legend().remove()

        if i != 2:
            plot.set(xlabel=None)
        if i != 1:
            plot.set(ylabel=None)

    handles, labels = plot.get_legend_handles_labels()
    fig.legend(loc='lower center', ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.25, 0.005))
    fig.set_figheight(2.5)
    plt.tight_layout()

    plt.savefig(figname)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--figname', type=str, default='result_1DLaplaceMix.pdf')
    args = parser.parse_args()
    main(args.figname)
