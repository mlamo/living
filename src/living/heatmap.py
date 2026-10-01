import numpy as np
import matplotlib.pyplot as plt
import time

import living.model as model


def plot_survival_probability_heatmap(eta: float):

    log10_rsh_list = np.linspace(-3, 1, 1000)
    log10_energ_list = np.linspace(2, 6, 1000)
    z_grid, en_grid = np.meshgrid(log10_rsh_list, log10_energ_list)

    start_time = time.time()
    psv_grid = model.survival_probability(10**en_grid, eta, 10**z_grid)
    end_time = time.time()

    total_time = end_time - start_time

    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(10**z_grid, 10**en_grid, psv_grid, cmap="RdBu", vmin=0.0, vmax=1.0, shading="auto")
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Survival Probability", fontsize=15)

    ax.set_xlabel(r"Source redshift z", fontsize=15)
    ax.set_ylabel(r"Neutrino energy [GeV]", fontsize=15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.annotate(
        rf"$\eta={eta:g}$",
        xy=(1, 1),
        xytext=(-10, -10),
        xycoords="axes fraction",
        textcoords="offset points",
        color="white",
        ha="right",
        va="top",
        fontsize=13,
    )

    plt.savefig(f"outputs/fig/survival_probs/psv_heatmap_eta{eta:.0e}.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    for eta in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        plot_survival_probability_heatmap(eta)
