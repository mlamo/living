from living.inputs.icecube import ICHESEDataset
from living.inputs.dataset import Dataset
from living.inputs.grb import GRBCatalog
import living.statistics as ast

import argparse
import line_profiler
import numpy as np
import os
import tomli


def prepare_inputs(dataset_class: type, config: dict) -> tuple[GRBCatalog, Dataset]:
    """Prepares GRB catalog and dataset for analysis.

    Loads GRB catalog from CSV file and computes probability density functions for the dataset.

    Args:
        dataset_class (type): The class of the dataset to instantiate.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        tuple: A tuple containing:
            - grbs (GRBCatalog): Filtered GRB catalog
            - dataset (Dataset): Dataset with computed PDFs
    """
    grbs = GRBCatalog()
    grbs.load_csv(config["grbs"]["catalog"])

    os.makedirs(config["output"]["folder_pdfs"], exist_ok=True)
    dataset = dataset_class(**config["neutrinos"]["dataset_args"], grbs=grbs)
    dataset.compute_pdfs(config["output"]["folder_pdfs"])

    return grbs, dataset


@line_profiler.profile
def compute_one_ts(
    dataset: Dataset,
    grbs: GRBCatalog,
    config: dict,
    scramble_grbs: bool = True,
    scramble_data: bool = True,
    fsig: float = 0,
    nsig: int = 0,
    eta: float = 0,
) -> tuple[float, float]:
    """Computes test statistic for one pseudo-experiment.

    Args:
        dataset (Dataset): Dataset containing background and signal PDFs.
        grbs (GRBCatalog): GRB catalog to use.
        config (dict): Dictionary containing all analysis parameters.
        scramble_grbs (bool, optional): Whether to scramble GRB times. Defaults to True.
        scramble_data (bool, optional): Whether to scramble data times. Defaults to True.
        fsig (float, optional): Fraction of signal events to inject. Defaults to 0.
        nsig (int, optional): Number of signal events to inject. Defaults to 0.
        eta (float, optional): LIV scaling parameter. Defaults to 0.

    Returns:
        tuple: Test statistic values (TS20, TS21) for hypothesis testing.
    """
    if scramble_grbs:
        grbs_pe = grbs.scramble()
    else:
        grbs_pe = grbs

    if (fsig is not None or nsig is not None) and eta > 0:
        data_pe = dataset.inject_signal(grbs_pe, eta, config, fsig, nsig)
    elif scramble_data:
        data_pe = dataset.scramble()
    else:
        data_pe = dataset.data

    events = data_pe.events(dataset, grbs_pe)
    res = ast.test_statistic(events, dataset, grbs_pe, config, return_bestfit=True)
    return res


def run_pe_signalinj(dataset: Dataset, grbs: GRBCatalog, fsig: float, eta: float, config: dict) -> None:
    """Runs pseudo-experiment with signal injection and fits parameters.

    Injects signal events into scrambled data, performs likelihood fits,
    and generates plots and output files for analysis.

    Args:
        grbs (GRBCatalog): GRB catalog to use for injection.
        dataset (Dataset): Dataset containing background and signal PDFs.
        fsig (float): Fraction of signal events to inject.
        eta (float): LIV scaling parameter for injected signal.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        float: best-fit fraction of signal events
        float: best-fit LIV scale
        float: p-value
    """
    grbs_pe = grbs.scramble()

    data_pe = dataset.inject_signal(grbs_pe, eta, config, fsig)
    events = data_pe.events(dataset, grbs_pe)

    ts20, _, bf_fsig, bf_eta = ast.test_statistic(events, dataset, grbs_pe, config, return_bestfit=True)
    tsbkg20 = np.loadtxt(os.path.join(config["output"]["folder_pes"], "pe_bkg.csv"))[:, 0]
    pvalue = np.count_nonzero(tsbkg20 >= ts20) / len(tsbkg20)

    ast.plot_loglikelihoodratio(
        os.path.join(config["output"]["folder_fig"], "siginj", f"likelihood_fsig{fsig:.2e}_eta{eta:.2e}.png"),
        events,
        dataset,
        grbs_pe,
        config,
        (0, 3 * fsig),
        (0, 3 * eta),
        log=False,
        zmax=5,
    )
    ast.plot_loglikelihoodratio(
        os.path.join(config["output"]["folder_fig"], "siginj", f"posterior_fsig{fsig:.2e}_eta{eta:.2e}.png"),
        events,
        dataset,
        grbs_pe,
        config,
        (0, 3 * fsig),
        (0, 3 * eta),
        log=False,
        bayesian=True,
    )
    ast.plot_deltat_energy_plot(
        os.path.join(config["output"]["folder_fig"], "siginj", f"deltat-energy_fsig{fsig:.2e}_eta{eta:.2e}.pdf"),
        events,
        dataset,
        grbs_pe,
        eta,
        config,
        choice="dist",
    )

    return bf_fsig, bf_eta, pvalue


def run_ts() -> None:
    """Main function to run test statistic analysis.

    Computes test statistics for multiple pseudo-experiments with specified
    signal injection parameters and saves results to files.

    If fsig != 0 or nsig != 0, the signal hypothesis is considered, with a number of signal events N_s:
        fsig != 0 → N_s ~ Poisson(fsig * Ndata)
        nsig != 0 → N_s = nsig

    Command line arguments:
        --npes: Number of pseudo-experiments to generate (default: 100)
        --fsig: Fraction of signal events to inject (default: 0)
        --nsig: Number of signal events to inject (default: 0)
        --eta: LIV scaling parameter (default: 0)
        --config: Path to configuration file
    """
    parser = argparse.ArgumentParser(description="Compute the TS for scrambled data or data with some signal injection.")
    parser.add_argument("--npes", type=int, default=100, help="Number of PEs to be generated. Default: 100.")
    parser.add_argument("--fsig", type=float, default=0, help="Injected fraction of signal events. Default: 0.")
    parser.add_argument("--nsig", type=int, default=0, help="Number of injected signal events. Default: 0.")
    parser.add_argument("--eta", type=float, default=0, help="Injected eta. Default: 0.")
    parser.add_argument("--config", type=str, help="Path to configuration file.")
    args = parser.parse_args()

    fsig, nsig = 0, None
    if args.fsig > 0:
        fsig = args.fsig
        nsig = None
    elif args.nsig > 0:
        fsig = None
        nsig = args.nsig

    with open(args.config, "rb") as f:
        config = tomli.load(f)

    grbs, dataset = prepare_inputs(globals()[config["neutrinos"]["dataset_class"]], config)

    results = []
    for _ in range(args.npes):
        res = compute_one_ts(dataset, grbs, config, fsig=fsig, nsig=nsig, eta=args.eta)
        if np.isnan(res[0]):
            continue
        results.append(res)

    outfolder = config["output"]["folder_pes"]
    os.makedirs(outfolder, exist_ok=True)
    if fsig is not None and args.eta > 0:
        outfile = f"{outfolder}/pe_sig_fsig{fsig:.2e}_eta{args.eta:.2e}.csv"
    elif nsig is not None and args.eta > 0:
        outfile = f"{outfolder}/pe_sig_nsig{nsig:d}_eta{args.eta:.2e}.csv"
    else:
        outfile = f"{outfolder}/pe_bkg.csv"
    with open(outfile, "ab") as f:
        np.savetxt(f, results)
