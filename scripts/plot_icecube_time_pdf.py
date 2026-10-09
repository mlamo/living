from living.run import prepare_inputs
from living.inputs.grb import GRB, GRBCatalog
from living.inputs.icecube import ICHESEDataset
import living.model as am

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tomli
from collections import defaultdict


def plot_time_pdf(configfile: str, eta: float, erecos: list):

    with open(configfile, "rb") as f:
        config = tomli.load(f)
    os.makedirs(config['output']['folder_fig'], exist_ok=True)

    grbs, dataset = prepare_inputs(dataset_class=globals()[config["neutrinos"]["dataset_class"]], config=config)
    grb = GRB("Test", 57000.5, 0, 0, 0.1)
    grbc = GRBCatalog()
    grbc.add_grb(grb)
    for d in dataset._data.values():
        d.compute_all_distances(grbc)

    y = defaultdict(dict)
    x = np.linspace(grb.time - 1/24, grb.time + 1, 100001)
    lsts = np.zeros_like(x)
    for samplename in ("track", "shower"):
        for ereco in erecos:
            energy, types = np.full(len(x), np.log10(ereco)), np.full(len(x), samplename)
            y[samplename][ereco] = am.time_pdf(eta, x, energy, types, lsts, dataset, grb, config)
            y[samplename][ereco] = np.nan_to_num(y[samplename][ereco], nan=0)

    outfig = os.path.join(config["output"]["folder_pdfs"], f"sig_time_pdf_eta{eta:.0e}_nliv{config['evaluation']['liv_order']}.pdf")

    plt.close("all")
    fig, ax = plt.subplots()
    handles = []
    for samplename, color in zip(["track", "shower"], ["#F7A278", "#6DD3CE"]):
        for ereco, ls in zip(erecos, ["solid", "dashed", "dashdot"]):
            ax.plot(x-grb.time, y[samplename][ereco] / y[samplename][ereco].max(), color=color, ls=ls)
    for samplename, color in zip(["Tracks", "Cascades"], ["#F7A278", "#6DD3CE"]):
        handles.append(matplotlib.patches.Patch(facecolor=color, label=samplename))
    for ereco, ls in zip(erecos, ["solid", "dashed", "dashdot"]):
        handles.append(matplotlib.lines.Line2D([], [], color="black", ls=ls, label=rf"$E_{{\text{{reco}}}} = {ereco/1e3:g}$ TeV"))
    label = [
        rf"$n = {config['evaluation']['liv_order']}$",
        rf"z={grb.redshift:.1f}",
        rf"$\eta={eta:.0f}$",
    ]
    handles.append(matplotlib.patches.Patch(color='none', label=", ".join(label)))
    ax.set_xlabel(r"MJD - MJD$_{\text{GRB}}$", fontsize=15)
    ax.set_ylabel(r"PDF (a.u.)", fontsize=13)
    ax.set_xscale("symlog", linthresh=10/(24*60))
    ax.set_xticks([-1/24, -10/(24*60), -1/(24*60), 1/(24*60), 10/(24*60), 1/24, 1], ["-1h", "-10'", "-1'", "1'", "10'", "1h", "1d"], minor=False)
    ax.set_xticks(np.append(np.append(1/24 * np.arange(1, 25), 1/(24*60) * np.arange(-60, 61, 10)), 1/(24*60) * np.arange(-10, 11)), minor=True)
    ax.set_xlim(-1/24, 1)
    ax.axvline((1+grb.redshift)*config["grbs"]["intrinsic_time_delay"][0] / 86400, color="k", ls="dotted")
    ax.axvline((1+grb.redshift)*config["grbs"]["intrinsic_time_delay"][1] / 86400, color="k", ls="dotted")
    ax.legend(handles=handles, loc="lower center", fontsize=11, ncol=3, bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    fig.savefig(outfig)


if __name__ == "__main__":

    for eta in (10, 20, 100,):
        plot_time_pdf("inputs/config/icecube_order1.toml", eta, [1e3, 2e4, 1e5])
    for eta in (1e6,):
        plot_time_pdf("inputs/config/icecube_order2.toml", eta, [1e6, 5e6])
