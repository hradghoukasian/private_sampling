import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# LaTeX-like formatting (MathText)
matplotlib.rcParams.update({
    'text.usetex': False,            # True will not work in Colab
    'mathtext.fontset': 'stix',      # Use LaTeX-style fonts
    'font.family': 'STIXGeneral',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 6)
})

# Load your data
data = np.load("LaplaceMixture_densities.npz")
original_densities = data["original"]
global_pure = data["global_pure"]
local_pure = data["local_pure"]
global_gdp = data["global_gdp"]
local_gdp = data["local_gdp"]

# Setup grid
abs_range = 2
x = np.linspace(-abs_range, abs_range, 500)
y = np.linspace(-abs_range, abs_range, 500)
xticks = [-2, 0, 2]
yticks = [-2, 0, 2]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Titles for each subplot
titles = [
    r'Original', r'Global Pure', r'Local Pure',
    r'Original', r'Global GDP', r'Local GDP'
]

# Densities to plot
densities = [
    original_densities, global_pure, local_pure,
    original_densities, global_gdp, local_gdp
]

# Loop through subplots
for i in range(2):
    for j in range(3):
        idx = i * 3 + j
        ax = axes[i, j]
        ax.imshow(densities[idx], extent=(-abs_range, abs_range, -abs_range, abs_range),
                  cmap='viridis', aspect='auto')
        ax.set_title(titles[idx], fontsize=20)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks if j == 0 else [])  # Show y-ticks only on first column
        

# Layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("LaplaceMixture_2x3Comparison.pdf", bbox_inches='tight')
plt.savefig("LaplaceMixture_2x3Comparison.eps", bbox_inches='tight')
plt.show()

