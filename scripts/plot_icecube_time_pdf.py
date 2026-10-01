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
    dt_max = am.time_delay_liv(eta, grb.Dn(config["evaluation"]["liv_order"]), 2*max(erecos), config["evaluation"]["liv_order"]) + (1+grb.redshift) * 400 / 86400
    if config["evaluation"]["liv_order"] == 1:
        x = np.linspace(grb.time - 0.02, grb.time + dt_max, 10001)
    else:
        x = np.linspace(grb.time - 0.01, grb.time + dt_max, 10001)
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
        for ereco, ls in zip(erecos, ["solid", "dashed"]):
            ax.plot(x-grb.time, y[samplename][ereco], color=color, ls=ls)
    for samplename, color in zip(["Track sample", "Cascade sample"], ["#F7A278", "#6DD3CE"]):
        handles.append(matplotlib.patches.Patch(facecolor=color, label=samplename))
    for ereco, ls in zip(erecos, ["solid", "dashed"]):
        handles.append(matplotlib.lines.Line2D([], [], color="black", ls=ls, label=rf"$E_{{\text{{reco}}}} = {ereco/1e3:g}$ TeV"))
    label = [
        rf"z={grb.redshift:.1f}",
        rf"$\eta={eta:.0f}$",
        rf"$n_{{\text{{LIV}}}} = {config['evaluation']['liv_order']}$"
    ]
    ax.annotate("\n".join(label), xy=(0.95, 0.65), xycoords="axes fraction", fontsize=13, va="top", ha="right")
    ax.set_xlabel(r"MJD - MJD$_{\text{GRB}}$", fontsize=15)
    ax.set_ylabel(r"PDF [day$^{-1}$]", fontsize=13)
    ax.axvline((1+grb.redshift)*config["grbs"]["intrinsic_time_delay"][0] / 86400, color="k", ls="dotted")
    ax.axvline((1+grb.redshift)*config["grbs"]["intrinsic_time_delay"][1] / 86400, color="k", ls="dotted")
    ax.legend(handles=handles, loc="upper right", fontsize=13)
    plt.tight_layout()
    fig.savefig(outfig)


if __name__ == "__main__":

    for eta in (0.1,1,20):
        plot_time_pdf("inputs/config/icecube_order1.toml", eta, [6e4, 2e5])
    for eta in (1e6, 1e7):
        plot_time_pdf("inputs/config/icecube_order2.toml", eta, [5e5, 1e6])