from algae.run import prepare_inputs
from algae.inputs.grb import GRB, GRBCatalog
from algae.inputs.icecube import ICHESEDataset
import algae.statistics as ast
import algae.model as am

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tomli
from collections import defaultdict
from colorama import Fore, Back, Style


def run_fit() -> None:

    parser = argparse.ArgumentParser(description="Run several checks.")
    parser.add_argument("--config", type=str, help="Path to configuration file.")
    parser.add_argument("--unblind", action="store_true", help="Unblind the analysis.")

    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomli.load(f)
    os.makedirs(config['output']['folder_fig'], exist_ok=True)
    os.makedirs(os.path.join(config['output']['folder_fig'], "siginj"), exist_ok=True)

    grbs, dataset = prepare_inputs(dataset_class=globals()[config["neutrinos"]["dataset_class"]], config=config)
    if not args.unblind:
        grbs = grbs.scramble()
        data = dataset.scramble()
    else:
        # while input(f"{Fore.WHITE}{Back.RED}You are about to unblind the analysis. Are you SURE?{Style.RESET_ALL} ({Fore.GREEN}Y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}) → ") != "Y":
        #     continue
        data = dataset.data
    events = data.events(dataset, grbs)
    
    pre = 'UNBLINDED_' if args.unblind else ''

    # Data TS
    tsdata20, tsdata21 = ast.test_statistic(events, dataset, grbs, config)
    print(f"TS(data,20) = {tsdata20:.2e}")
    print(f"TS(data,21) = {tsdata21:.2e}")
    # Background TS distribution
    outfile = os.path.join(config['output']['folder_pes'], "pe_bkg.csv")
    ts20 = np.loadtxt(outfile)[:,0]
    ts21 = np.loadtxt(outfile)[:,1]
    # p-value computation
    pvalue20 = np.count_nonzero(ts20 >= tsdata20) / len(ts20)
    pvalue21 = np.count_nonzero(ts21 >= tsdata21) / len(ts21)
    print(f"p-value (2-0): {100*pvalue20:.2e}%")
    print(f"p-value (2-1): {100*pvalue21:.2e}%")

    # Plot TS distribution
    _, ax = plt.subplots()
    h, bins = np.histogram(ts20, bins=30)
    ax.stairs(h, bins, fill=True, edgecolor="#9a9a9a", facecolor="#bcbcbc", lw=2, label=r"Background")
    ax.axvline(tsdata20, ls="dashed", color="black", label=rf"Observed ($p={100*pvalue20:.1f}\%$)")
    ax.set_xlabel("Test Statistic", fontsize=15)
    ax.set_ylabel("# of PEs", fontsize=15)
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc="best", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["folder_fig"], f"{pre}ts-bkg.pdf"), dpi=300)

    # Best-fit and contour
    bf_fsig, bf_eta = ast.bestfit(events, dataset, grbs, config)
    bf_flux = None
    if hasattr(dataset, "psflux_to_nsig"):
        bf_flux = bf_fsig * 1/dataset.psflux_to_nsig(spectral_index=2.52) * len(dataset.data)
    print(f"H2 best-fit point =", (bf_fsig, bf_eta, bf_flux))

    bf_fsig_H1 = ast.bestfit_H1(events, dataset, grbs, config)
    print(f"H1 best-fit point =", bf_fsig_H1)
    c = ast.contour_profile_eta(events, dataset, grbs, config)
    print(f"1-2-3sig interval for eta =", c)

    # Top matches
    s = ast.signal_term(bf_eta, events, dataset, grbs, config, keep_split=True)
    sum_s = np.sum(s, axis=0)
    b = ast.background_term(events, dataset, config)
    idx_best = np.unravel_index(np.flipud(np.argsort(s.flatten())[-10:]), s.shape)
    for i, (igrb, ievt) in enumerate(zip(*idx_best)):
        if s[igrb][ievt] == 0:
            break
        g = grbs._grbs[igrb]
        e = dataset.data._events.iloc[ievt]
        print(f"\n======================\nTOP {i+1}\n======================\n")
        print(g)
        print({k: v for k, v in e.items() if "alpha" not in k and "sigang" not in k})
        emed = dataset.evaluate_signal_energy_icdf(dataset.data._events.iloc[[ievt]].energy.to_numpy(), np.array([0.5]), dataset.data._events.iloc[[ievt]].type.to_numpy())[0]
        dtliv = am.time_delay_liv(bf_eta, g.Dn(config["evaluation"]["liv_order"]), emed, config["evaluation"]["liv_order"])
        print(f"DeltaT = {e.time - g.time:.2f} days ; DeltaT_exp,LIV(E=Emed) = {dtliv:.2f} days")
        print(f"Angular distance = {e[f'alpha_{g.name}']:.2f} degrees")
        print(f"Signal term = {s[igrb][ievt]:.2e}/{sum_s[ievt]:.2e}s ; Background term = {b[ievt]} ; S/B = {sum_s[ievt]/b[ievt]:.2e}")
        print(f"Best eta = {np.abs(e.time - g.time)/dtliv * bf_eta:.2e}")
    print(f"\n======================\n")

    # Plot 
    range_fsig_lin = (0, 3*bf_fsig)
    range_fsig_log = (1e-3, 1e-1)
    if config["evaluation"]["liv_order"] == 1:
        range_eta_lin = (0, 3*bf_eta)
        range_eta_log = (1e0, 1e4)
    else:
        if bf_fsig == 0:
            bf_fsig = 1e-2
        if bf_eta == 0:
            bf_eta = 1e6
        range_eta_lin = (0, 3*bf_eta)
        range_eta_log = (1e5, 1e10)
    ast.plot_loglikelihoodratio(os.path.join(config['output']['folder_fig'], f"{pre}likelihood_log.pdf"), events, dataset, grbs, config, range_fsig_log, range_eta_log, log=True, bayesian=False, zmax=2)
    if hasattr(dataset, "psflux_to_nsig"):
        conv = 1/dataset.psflux_to_nsig(spectral_index=2) * len(dataset.data)
        ast.plot_loglikelihoodratio(os.path.join(config['output']['folder_fig'], f"{pre}likelihood-flux_log.pdf"), events, dataset, grbs, config, range_fsig_log, range_eta_log, log=True, bayesian=False, zmax=3, conv_fsig=conv)
    ast.plot_deltat_energy_plot(os.path.join(config['output']['folder_fig'], f"{pre}deltat_energy.pdf"), events, dataset, grbs, bf_eta, config, choice="all")


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

    # run_fit()

    for eta in (0.1,1,20):
        plot_time_pdf("inputs/config/icecube_order1.toml", eta, [6e4, 2e5])
    for eta in (1e6, 1e7):
        plot_time_pdf("inputs/config/icecube_order2.toml", eta, [5e5, 1e6])
