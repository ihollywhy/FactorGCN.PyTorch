import os
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
sns.set()


def plot_corr(data, save=''):
    root = "./data/figs/correlations"
    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    plt.savefig(os.path.join(root, save), dpi=800)
    plt.close()