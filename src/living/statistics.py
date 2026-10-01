import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.interpolate as si
import scipy.stats as sst
import scipy.optimize as so
from functools import partial

from living.inputs.dataset import Dataset
from living.inputs.grb import GRBCatalog
from living.model import time_pdf, time_delay_liv, invert_H0


def signal_term(eta: float, events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict, keep_split: bool = False) -> np.ndarray:
    """Signal term used in the likelihood defined as sum_{g in GRBs} S_{i, g}, where
        S_{i, g} = S_ang(Angle(GRB, evt i); E_{rec,i}) * S_time(t_{i}; eta, E_{rec,i})
    If energy_term is True, an additional term S_ereco(E_{rec,i}; gamma) is added.

    Args:
        eta (float): LIV scaling.
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalogue.
        config (dict): Dictionary containing all analysis parameters.
        keep_split (bool, optional): whether the terms for each GRBs should be kept split or summed. Defaults to False.

    Returns:
        np.ndarray: value of signal term for each event (dim = num.evts). If keep_split=True, it is not summed over GRBs (dim = num.GRBs x num.evts).
    """
    n_events = len(events)
    n_grbs = len(grbs)

    cone_sizes = np.array([dataset.cone_size[t] for t in events.type])
    if not np.all(np.array(np.isfinite(list(dataset.cone_size.values())))):
        filter_matrix = None
    else:
        alpha_cols = [f"alpha_{g.name}" for g, _ in grbs]
        filter_matrix = [events[c] < cone_sizes for c in alpha_cols]
    ev_time, ev_energy, ev_type, ev_lst = events["time"], events["energy"], events["type"], events["lst"]

    signal_terms = np.zeros((n_grbs, n_events), dtype=np.float64)

    for idx, (grb, weight) in enumerate(grbs):
        if filter_matrix:
            filt = filter_matrix[idx]
            if not np.any(filt):
                continue
            evf_time, evf_energy, evf_type, evf_lst = ev_time[filt], ev_energy[filt], ev_type[filt], ev_lst[filt]
            sig_ang = events[f"sigang_{grb.name}"][filt]
            sig_time = time_pdf(eta, evf_time, evf_energy, evf_type, evf_lst, dataset, grb, config)
            signal_terms[idx, filt] = weight * sig_ang * sig_time
        else:
            sig_ang = events[f"sigang_{grb.name}"]
            sig_time = time_pdf(eta, ev_time, ev_energy, ev_type, ev_lst, dataset, grb, config)
            signal_terms[idx] = weight * sig_ang * sig_time

    if config["evaluation"]["include_energy_term"]:
        signal_terms *= events["sigereco"]

    if keep_split:
        return signal_terms
    return np.sum(signal_terms, axis=0)


def background_term(events: np.recarray, dataset: Dataset, config: dict) -> np.ndarray:
    """Background term used in the likelihood defined as
        B_{i} = B_ang(Angle(GRB, evt i); E_{rec,i}) * B_time(t_{i})
    If energy_term is True, an additional term B_ereco(E_{rec,i}) is added.

    Args:
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        np.ndarray: Value of background term for each event (dim = num.evts).
    """
    background_term = dataset.evaluate_background_angular_pdf(events) * dataset.evaluate_background_time_pdf(events)
    if config["evaluation"]["include_energy_term"]:
        background_term *= dataset.evaluate_signal_ereco_pdf(events.energy, events.type)
    return background_term


def loglikelihood(fsig: np.ndarray, sig: np.ndarray, bkg: np.ndarray) -> np.ndarray:
    """Computes the log-likelihood for given signal fractions.

    Args:
        fsig (np.ndarray): Signal fraction(s) to evaluate.
        sig (np.ndarray): Signal terms for each event.
        bkg (np.ndarray): Background terms for each event.

    Returns:
        np.ndarray: Log-likelihood values for each fsig value.
    """
    fsig = np.array(fsig)
    return np.sum(np.log(fsig[..., np.newaxis] * sig[np.newaxis, ...] + (1 - fsig[..., np.newaxis]) * bkg[np.newaxis, ...]), axis=1)


def derivative_loglikelihood(fsig: float, sig: np.ndarray, bkg: np.ndarray) -> np.ndarray:
    """Computes the derivative of the log-likelihood with respect to fsig.

    Args:
        fsig (float): Signal fraction at which to evaluate derivative.
        sig (np.ndarray): Signal terms for each event.
        bkg (np.ndarray): Background terms for each event.

    Returns:
        np.ndarray: Derivative values for each fsig value.
    """
    fsig = np.array(fsig)
    return np.sum(
        (sig[np.newaxis, ...] - bkg[np.newaxis, ...])
        / (fsig[..., np.newaxis] * sig[np.newaxis, ...] + (1 - fsig[..., np.newaxis]) * bkg[np.newaxis, ...]),
        axis=1,
    )


def test_statistic(events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict, return_bestfit: bool = False) -> tuple[float, float]:
    """Computes test statistics for hypothesis testing.

    Computes likelihood ratio test statistics comparing:
    - H2 vs H0: Signal+background vs background only
    - H2 vs H1: Signal+background vs signal+background with fsig=0

    Args:
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        config (dict): Dictionary containing all analysis parameters.
        return_bestfit (bool, optional): whether the function also returns the best-fit values for fsig and eta. Defaults to False.

    Returns:
        tuple: Test statistic values (TS20, TS21) for H2 vs H0 and H2 vs H1.
    """
    min_fsig = max(1 / len(events), config["evaluation"]["range_fit_fsig"][0])
    max_fsig = config["evaluation"]["range_fit_fsig"][1]
    bkg = background_term(events, dataset, config)

    def func(log10eta):
        sig = signal_term(10**log10eta, events, dataset, grbs, config)
        best_fsig = np.clip(so.root_scalar(partial(derivative_loglikelihood, sig=sig, bkg=bkg), x0=min_fsig).root[0], min_fsig, max_fsig)
        return -loglikelihood(best_fsig, sig, bkg)

    # Compute best-fit signal fraction under H1 hypothesis (fsig free, eta=0)
    sig_H1 = signal_term(0, events, dataset, grbs, config)
    best_fsig_H1 = np.clip(so.root_scalar(partial(derivative_loglikelihood, sig=sig_H1, bkg=bkg), x0=min_fsig).root[0], min_fsig, max_fsig)

    # Compute all log-likelihoods and test statistics
    res = so.minimize_scalar(func, bounds=config["evaluation"]["range_fit_log10eta"])
    lik_h0 = loglikelihood(0, np.zeros_like(bkg), bkg)[0]
    lik_h1 = max(lik_h0, loglikelihood(best_fsig_H1, sig_H1, bkg)[0])
    lik_h2 = max(lik_h0, -res.fun[0])
    llr_20 = 2 * (lik_h2 - lik_h0)
    llr_21 = 2 * (lik_h2 - lik_h1)

    # If requested, also return best-fit parameters under H2 hypothesis
    if return_bestfit:
        best_eta = float(10 ** res.x[0])
        sig = signal_term(best_eta, events, dataset, grbs, config)
        best_fsig = float(np.clip(so.root_scalar(partial(derivative_loglikelihood, sig=sig, bkg=bkg), x0=0).root[0], min_fsig, max_fsig))
        return llr_20, llr_21, best_fsig, best_eta

    return llr_20, llr_21


def bestfit(events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict) -> tuple[float, float]:
    """Finds best-fit signal fraction and eta parameters under H2 hypothesis (w/ neutrinos from GRBs, w/ LIV).

    Args:
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        tuple: Best-fit parameters (best_fsig, best_eta).
    """

    min_fsig = max(1 / len(events), config["evaluation"]["range_fit_fsig"][0])
    max_fsig = config["evaluation"]["range_fit_fsig"][1]
    bkg = background_term(events, dataset, config)

    def func(log10eta):
        sig = signal_term(10**log10eta, events, dataset, grbs, config)
        best_fsig = np.clip(so.root_scalar(partial(derivative_loglikelihood, sig=sig, bkg=bkg), x0=0).root[0], min_fsig, max_fsig)
        return -loglikelihood(best_fsig, sig, bkg)

    lik_h0 = loglikelihood(0, np.zeros_like(bkg), bkg)
    res = so.minimize_scalar(func, bounds=config["evaluation"]["range_fit_log10eta"])
    # If the best-fit likelihood is not better than H0, set best-fit parameters to 0
    if -res.fun[0] < lik_h0:
        best_eta = 0
        best_fsig = 0
    else:
        best_eta = float(10 ** res.x[0])
        sig = signal_term(best_eta, events, dataset, grbs, config)
        best_fsig = float(np.clip(so.root_scalar(partial(derivative_loglikelihood, sig=sig, bkg=bkg), x0=0).root[0], min_fsig, max_fsig))
    return best_fsig, best_eta


def bestfit_H1(events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict) -> float:
    """Finds best-fit signal fraction under H1 hypothesis (w/ neutrinos from GRBs, no LIV).

    Args:
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        float: Best-fit signal fraction.
    """
    bkg = background_term(events, dataset, config)
    sig = signal_term(0, events, dataset, grbs, config)
    func = partial(derivative_loglikelihood, sig=sig, bkg=bkg)
    return np.clip(so.root_scalar(func, x0=0).root[0], *config["evaluation"]["range_fit_fsig"])


def contour_profile_eta(events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict, nsigmas: list[int] = [1, 2, 3]) -> np.ndarray:
    """Computes confidence intervals for eta parameter using likelihood profiling.

    Args:
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        config (dict): Dictionary containing all analysis parameters.
        nsigmas (list, optional): Confidence levels in sigma. Defaults to [1, 2, 3].

    Returns:
        np.ndarray: Confidence intervals [[eta_min_1sig, eta_max_1sig],
                                        [eta_min_2sig, eta_max_2sig],
                                        [eta_min_3sig, eta_max_3sig]].
    """
    bkg = background_term(events, dataset, config)

    def func(log10eta):
        sig = signal_term(10**log10eta, events, dataset, grbs, config)
        func2 = partial(derivative_loglikelihood, sig=sig, bkg=bkg)
        fsig_min = np.clip(so.root_scalar(func2, x0=0).root[0], *config["evaluation"]["range_fit_fsig"])
        return -loglikelihood(fsig_min, sig, bkg)

    res = so.minimize_scalar(func, bounds=config["evaluation"]["range_fit_log10eta"])
    best_eta = 10 ** res.x[0]
    best_fun = res.fun[0]

    levels = sst.chi2.isf(sst.norm.sf(nsigmas), 2)
    contours = []
    for level in levels:
        try:
            eta_min = so.root_scalar(lambda x: func(np.log10(x)) - (best_fun + level), bracket=(0, best_eta)).root
        except:
            eta_min = 0
        try:
            eta_max = so.root_scalar(lambda x: func(np.log10(x)) - (best_fun + level), bracket=(best_eta, 1e100)).root
        except:
            eta_max = 1e100
        contours.append([eta_min, eta_max])
    return np.array(contours)


def compute_fsig_threshold(infolder: str, outfile: str, eta_fixed: float, ndata: float, alpha: float = sst.norm.cdf(3), beta: float = 0.50):
    """Compute the value of fsig needed to get Prob(TS > TS_thr; fsig) = beta,
    where TS_thr is defined as the solution of Prob(TS <= TS_thr; bkg) = alpha
    """

    ts_bkg = np.loadtxt(f"{infolder}/pe_bkg.csv")[:, 0]
    ts_threshold = np.percentile(ts_bkg, 100 * alpha)

    # Load test statistic distributions for different numbers of injected signal events (nsig) and fixed eta.
    files = glob.glob(f"{infolder}/pe_sig_*_eta{eta_fixed:.2e}.csv")
    ts = {0: ts_bkg}
    for f in files:
        match = re.search(r"nsig([-+]?\d+)", f)
        if not match:
            continue
        nsig = int(match.group(1))
        ts[nsig] = np.loadtxt(f)[:, 0]
    nsig_max = max(list(ts.keys()))

    # Define a function to compute the power for a given fsig value, using the test statistic distributions.
    def get_power(fsig: float, ntoys: int = 10000):
        nsig, nit = np.unique(sst.poisson.rvs(fsig * ndata, size=ntoys), return_counts=True)
        toys_ts = []
        for _nsig, _nit in zip(nsig, nit):
            if _nsig > nsig_max:
                # print(f"Missing simulations for nsig={_nsig}, will use as proxy simulations for nsig={nsig_max}")
                _nsig = nsig_max
            toys_ts = np.append(toys_ts, np.random.choice(ts[_nsig], size=_nit))
        return np.count_nonzero(toys_ts > ts_threshold) / ntoys

    # Use a root-finding algorithm to find the fsig value that gives the desired power.
    try:
        fsig_threshold = so.bisect(lambda fsig: get_power(fsig) - beta, 0, 10 / ndata)
    except:
        return np.nan

    # Plot the power curve as a function of fsig, and save the figure.
    plt.close("all")
    plot_fsig = np.linspace(0, 4 * fsig_threshold, 201)
    plot_thr = np.array([get_power(_fsig) for _fsig in plot_fsig])
    plt.plot(plot_fsig, 100 * plot_thr)
    plt.axvline(fsig_threshold, color="red", ls="dashed", label=rf"$f_{{\text{{sig}}}} = {fsig_threshold:.2e}$")
    plt.axhline(100 * beta, color="red", ls="dashed")
    plt.xlabel(r"$f_{\text{sig}}$", fontsize=15)
    plt.ylabel(r"Prob$(TS > TS_{\text{thr}})$", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    return fsig_threshold


def compute_fsig_upper_limit(infolder: str, outfile: str, eta_fixed: float, ndata: float, ts_observed: float, cl: float = 0.90):

    ts_bkg = np.loadtxt(f"{infolder}/pe_bkg.csv")[:, 0]

    # Load test statistic distributions for different numbers of injected signal events (nsig) and fixed eta.
    files = glob.glob(f"{infolder}/pe_sig_*_eta{eta_fixed:.2e}.csv")
    ts = {0: ts_bkg}
    for f in files:
        match = re.search(r"nsig([-+]?\d+)", f)
        if not match:
            continue
        nsig = int(match.group(1))
        ts[nsig] = np.loadtxt(f)[:, 0]
    nsig_max = max(list(ts.keys()))

    # Define a function to compute the power for a given fsig value, using the test statistic distributions.
    def get_power(fsig: float, ntoys: int = 10000):
        nsig, nit = np.unique(sst.poisson.rvs(fsig * ndata, size=ntoys), return_counts=True)
        toys_ts = []
        for _nsig, _nit in zip(nsig, nit):
            if _nsig > nsig_max:
                _nsig = nsig_max
            toys_ts = np.append(toys_ts, np.random.choice(ts[_nsig], size=_nit))
        return np.count_nonzero(toys_ts > ts_observed) / ntoys

    # Use a root-finding algorithm to find the fsig value that gives the desired power.
    try:
        fsig_ul = so.bisect(lambda fsig: get_power(fsig) - cl, 0, 10 / ndata)
    except:
        return np.nan

    # Plot the power curve as a function of fsig, and save the figure.
    plt.close("all")
    plot_fsig = np.linspace(0, 4 * fsig_ul, 201)
    plot_thr = np.array([get_power(_fsig) for _fsig in plot_fsig])
    plt.plot(plot_fsig, 100 * plot_thr)
    plt.axvline(fsig_ul, color="red", ls="dashed", label=rf"$f_{{\text{{sig}}}} = {fsig_ul:.2e}$")
    plt.axhline(100 * cl, color="red", ls="dashed")
    plt.xlabel(r"$f_{\text{sig}}$", fontsize=15)
    plt.ylabel(r"Prob$(TS > TS_{\text{observed}})$", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    return fsig_ul


def plot_loglikelihoodratio_1d(outfig: str, events: np.recarray, dataset: Dataset, grbs: GRBCatalog, config: dict, eta: float = 0) -> None:
    """Plot the profiled likelihood ratio as a function of fsig, for fixed eta.

    Args:
        outfig (str): Path to the output figure.
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        config (dict): Dictionary containing all analysis parameters.
        eta (float, optional): Values of eta to be profiled on. Defaults to 0.
    """

    fsigs = np.logspace(-4, np.log10(0.99), 1001)
    bkg = background_term(events, dataset, config)
    sig = signal_term(eta, events, dataset, grbs, config)
    loglike = loglikelihood(fsigs, sig, bkg)
    llr = -2 * (np.array(loglike) - np.max(loglike))

    ibestfit = np.nanargmin(llr)
    bestfit_fsig = fsigs[ibestfit]

    plt.close("all")
    plt.plot(fsigs, llr, color="blue")
    plt.axvline(bestfit_fsig, color="black")
    for nsigmas, ls in zip([1, 2, 3], ["solid", "dashed", "dotted"]):
        plt.axhline(sst.chi2.isf(sst.norm.sf(nsigmas), 1), color="red", ls=ls)
    plt.xlabel(r"$f_{\rm sig}$", fontsize=15)
    plt.ylabel("LLR")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(outfig, dpi=300)


def plot_loglikelihoodratio(
    outfig: str,
    events: np.recarray,
    dataset: Dataset,
    grbs: GRBCatalog,
    config: dict,
    xlim: tuple,
    ylim: tuple,
    log: bool = True,
    zmax: float = None,
    bayesian: bool = False,
    conv_fsig: float = 1,
) -> None:
    """Plot the 2D likelihood ratio as a function of fsig and eta.

    Args:
        outfig (str): Path to the output figure.
        events (np.recarray): list of neutrino events
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        eta (float, optional): Values of eta to be profiled on. Defaults to 0.
        energy_term (bool, optional): Whether the energy term should be added in the PDF. Defaults to True.
        log (bool, optional): Uses log scale for eta and fsif. Defaults to True.
        zmax (float, optional): Max likelihood value. Defaults to None.
        bayesian (bool, optional): Flag to switch to Bayesian contours instead of Wilks'. Defaults to False.
        conv_fsig (float, optional): Conversion factor for fsig to flux. Defaults to 1.
    """

    # Define grid of eta and fsig values to evaluate likelihood on
    if log:
        fsigs = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 601)
        etas = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 121)
    else:
        fsigs = np.linspace(*xlim, 601)
        etas = np.linspace(*ylim, 121)

    # Compute likelihood values on the grid
    bkg = background_term(events, dataset, config)
    loglike = []
    for eta in etas:
        sig = signal_term(eta, events, dataset, grbs, config)
        loglike.append(loglikelihood(fsigs, sig, bkg))
    llr = -2 * (np.array(loglike) - np.nanmax(loglike))

    # Compute profiled likelihoods and best-fit parameters
    llr_prof_eta = np.nanmin(llr, axis=1)
    llr_prof_fsig = np.nanmin(llr, axis=0)
    ibestfit = np.unravel_index(np.nanargmin(llr), shape=llr.shape)
    bestfit_eta, bestfit_fsig = etas[ibestfit[0]], fsigs[ibestfit[1]]

    # Compute contour levels for Wilks' theorem or Bayesian credible regions
    if bayesian:
        pvalues = 1 - 2 * sst.norm.sf([3, 2, 1])
        post = np.exp(-llr)
        post /= np.sum(post)
        isort = np.flipud(np.argsort(post.flatten()))
        f = si.interp1d(np.cumsum(post.flatten()[isort]), post.flatten()[isort], bounds_error=False, fill_value=(np.max(post), 0))
        levels = f(pvalues)
        post_eta = np.sum(post, axis=1)
        isort_eta = np.flipud(np.argsort(post_eta))
        levels_eta = si.interp1d(np.cumsum(post_eta[isort_eta]), post_eta[isort_eta], bounds_error=False, fill_value=(np.max(post_eta), 0))(pvalues)
        post_fsig = np.sum(post, axis=0)
        isort_fsig = np.flipud(np.argsort(post_fsig))
        levels_fsig = si.interp1d(np.cumsum(post_fsig[isort_fsig]), post_fsig[isort_fsig], bounds_error=False, fill_value=(np.max(post_fsig), 0))(
            pvalues
        )
    else:
        pvalues = 2 * sst.norm.sf([1, 2, 3])
        levels = sst.chi2.isf(pvalues, 2)
        levels_eta = sst.chi2.isf(pvalues, 1)
        levels_fsig = sst.chi2.isf(pvalues, 1)

    plt.close("all")
    fig, ax = plt.subplots(2, 3, figsize=(9, 7), width_ratios=[0.6, 2, 0.15], height_ratios=[0.6, 2])
    ax[0, 0].set_axis_off()
    # Profiled likelihood as a function of eta
    ax[1, 0].plot(post_eta if bayesian else llr_prof_eta, etas, color="blue")
    for l, ls in zip(levels_eta, ["dotted", "dashed", "solid"] if bayesian else ["solid", "dashed", "dotted"]):
        ax[1, 0].axvline(l, color="black", ls=ls)
    # Profiled likelihood as a function of fsig
    ax[0, 1].plot(fsigs * conv_fsig, post_fsig if bayesian else llr_prof_fsig, color="blue")
    for l, ls in zip(levels_fsig, ["dotted", "dashed", "solid"] if bayesian else ["solid", "dashed", "dotted"]):
        ax[0, 1].axhline(l, color="black", ls=ls)
    # 2D likelihood
    pcm = ax[1, 1].pcolormesh(
        fsigs * conv_fsig, etas, post if bayesian else llr, cmap="Blues_r", shading="nearest", vmax=5 * zmax if zmax is not None else None
    )
    ax[1, 1].scatter(bestfit_fsig * conv_fsig, bestfit_eta, s=100, marker="x", color="black")
    cs = ax[1, 1].contour(
        fsigs * conv_fsig,
        etas,
        post if bayesian else llr,
        levels=levels,
        colors="black",
        linestyles=["dotted", "dashed", "solid"] if bayesian else ["solid", "dashed", "dotted"],
    )

    handles = [
        matplotlib.lines.Line2D([], [], ls="none", marker="x", color="black", label=rf"Best fit"),
        matplotlib.lines.Line2D([], [], ls="solid", color="black", label=r"$1\sigma$"),
        matplotlib.lines.Line2D([], [], ls="dashed", color="black", label=r"$2\sigma$"),
        matplotlib.lines.Line2D([], [], ls="dotted", color="black", label=r"$3\sigma$"),
    ]

    # Set scales, limits and labels
    if log:
        ax[0, 1].set_xscale("log")
        ax[1, 0].set_yscale("log")
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_yscale("log")
    ax[0, 1].set_xlim(ax[1, 1].get_xlim())
    if zmax is not None:
        ax[0, 1].set_ylim((-0.1, zmax))
        ax[1, 0].set_xlim((-0.1, zmax))
    ax[1, 0].set_ylim(ax[1, 1].get_ylim())
    ax[0, 2].axis("off")
    ax[0, 1].yaxis.set_label_position("right")
    ax[0, 1].tick_params(axis="y", which="both", left=False, labelleft=False, right=True, labelright=True)
    ax[0, 1].tick_params(axis="x", which="both", labelbottom=False)
    ax[1, 1].tick_params(labelleft=False)
    cb = fig.colorbar(pcm, cax=ax[1, 2])
    if conv_fsig != 1:
        ax[1, 1].set_xlabel(r"$E^{2} {\rm d}N/{\rm d}E$ at 1 GeV [GeV cm$^{-2}$]", fontsize=15)
    else:
        ax[1, 1].set_xlabel(r"$f_{\rm sig}$", fontsize=15)
    ax[1, 0].set_ylabel(r"$\eta$", fontsize=15)
    if bayesian:
        if conv_fsig != 1:
            ax[0, 1].set_ylabel(r"$P_{\text{post}}$($E^{2} {\rm d}N/{\rm d}E$ at 1 GeV)", fontsize=15)
            cb.set_label(r"$P_{\text{post}}$($E^{2} {\rm d}N/{\rm d}E$ at 1 GeV, $\eta$)", fontsize=15)
        else:
            ax[0, 1].set_ylabel(r"$P_{\text{post}}(f_{\text{sig}})$", fontsize=15)
            cb.set_label(r"$P_{\text{post}}(f_{\text{sig}}, \eta)$", fontsize=15)
        ax[1, 0].set_xlabel(r"$P_{\text{post}}(\eta)$", fontsize=15)
    else:
        ax[0, 1].set_ylabel(r"$\mathcal{L}_{prof}$", fontsize=15)
        ax[1, 0].set_xlabel(r"$\mathcal{L}_{prof}$", fontsize=15)
        cb.set_label(r"$-2\log \mathcal{L}/\mathcal{L}_{\max}$", fontsize=15)
    ax[1, 1].legend(handles=handles, loc="upper right", fontsize=13)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=10)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=10)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=10)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=10)

    # Adjust layout and save figure
    plt.subplots_adjust(left=0.08, right=0.91, bottom=0.1, top=0.96, wspace=0.03, hspace=0.03)
    plt.savefig(outfig, dpi=300)
    plt.savefig(outfig.replace(".pdf", ".png"), dpi=300)


def plot_deltat_energy_plot(
    outfig: str, events: np.recarray, dataset: Dataset, grbs: GRBCatalog, eta: float, config: dict, choice: str = "closest"
) -> None:
    """Plot the 2D distribution of events in xy plane, where:
    - for each event, the closest GRB is used
    - x = (t_i - t_g) / integral_0^{z_g} [(1+z) / sqrt(Omegam + (1+z)³ Omegad) dz]
    - y = median energy of the neutrino event given its reconstructed energy

    Args:
        outfig (str): Path to the output figure.
        events (np.recarray): List of neutrino events.
        dataset (Dataset): Dataset object.
        grbs (GRBCatalog): GRB catalog for signal modeling.
        eta (float, optional): Values of eta to be profiled on. Defaults to 0.
        config (dict): Dictionary containing all analysis parameters.
        choice (str, optional): Pairing approach for each event. "closest": use closest GRB, "best": use GRB with highest signal PDF.
    """
    n_liv = config["evaluation"]["liv_order"]
    alpha_cols = [f"alpha_{g.name}" for g, _ in grbs]
    distances = np.array([events[c] for c in alpha_cols]).T
    sigang_cols = [f"sigang_{g.name}" for g, _ in grbs]
    sigangpdfs = np.array([events[c] for c in sigang_cols]).T

    filt_spos = None
    filt_injected = None
    if choice == "closest":
        # For each event, plot the closest GRB based on angular distance
        closest_grb_idx = np.argmin(distances, axis=1)
        delta_t_over_I1 = (events.time - grbs.times[closest_grb_idx]) / (grbs.Dns(n_liv)[closest_grb_idx] / invert_H0)
        etrue_median = dataset.evaluate_signal_energy_icdf(events.energy, np.full(len(events), 0.5), events.type)
        if "comment" in events.dtype.names:
            filt_injected = events.comment != ""
        # Same size for all events, since we are not weighting by signal PDF
        s = 10 * np.ones_like(etrue_median)
    elif choice == "best":
        # For each event, plot the GRB with the highest signal PDF
        s = signal_term(eta, events, dataset, grbs, config, keep_split=True)
        best_grb_idx = np.argmax(s, axis=0)
        delta_t_over_I1 = (events.time - grbs.times[best_grb_idx]) / (grbs.Dns(n_liv)[best_grb_idx] / invert_H0)
        etrue_median = dataset.evaluate_signal_energy_icdf(events.energy, np.full(len(events), 0.5), events.type)
        s = np.amax(s, axis=0)
        filt_s = s > 0
        delta_t_over_I1, etrue_median = delta_t_over_I1[filt_s], etrue_median[filt_s]
        if "comment" in events.dtype.names:
            filt_injected = events.comment[filt_s] != ""
        # Scale size with the signal PDF, but keep it within a reasonable range for plotting
        s = np.clip(50 * (s[filt_s] / np.nanmax(s)), 0.1, 50)
    elif choice == "dist":
        # For each event, plot the GRB with the highest signal PDF
        s = signal_term(eta, events, dataset, grbs, config, keep_split=True)
        best_grb_idx = np.argmax(s, axis=0)
        delta_t_over_I1 = (events.time - grbs.times[best_grb_idx]) / (grbs.Dns(n_liv)[best_grb_idx] / invert_H0)
        etrue_median = dataset.evaluate_signal_energy_icdf(events.energy, np.full(len(events), 0.5), events.type)
        filt_s = np.amax(s, axis=0) > 0
        s = sigangpdfs[np.arange(sigangpdfs.shape[0]), best_grb_idx]
        delta_t_over_I1, etrue_median = delta_t_over_I1[filt_s], etrue_median[filt_s]
        if "comment" in events.dtype.names:
            filt_injected = events.comment[filt_s] != ""
        # Scale size with the angular signal PDF, but keep it within a reasonable range for plotting
        s = np.clip(50 * (s[filt_s] / np.nanmax(s)), 0.1, 50)
    elif choice == "all":
        # For each event, plot all GRBs within 20 degrees
        s = signal_term(eta, events, dataset, grbs, config, keep_split=True)
        filt_spos = s > 0
        delta_t_over_I1 = (events.time[np.newaxis, :] - grbs.times[:, np.newaxis]) / (grbs.Dns(n_liv)[:, np.newaxis] / invert_H0)
        etrue_median = dataset.evaluate_signal_energy_icdf(events.energy, np.full(len(events), 0.5), events.type)
        etrue_median = np.tile(etrue_median, (len(grbs), 1))
        filt_dist = distances.T < 20
        # s = sigangpdfs.T[filt_dist].flatten()
        s = s[filt_dist].flatten()
        filt_spos, delta_t_over_I1, etrue_median = (
            filt_spos[filt_dist].flatten(),
            delta_t_over_I1[filt_dist].flatten(),
            etrue_median[filt_dist].flatten(),
        )
        if "comment" in events.dtype.names:
            filt_injected = np.tile(events.comment != "", (len(grbs), 1))[filt_dist].flatten()
        # Scale size with the angular signal PDF, but keep it within a reasonable range for plotting
        s = np.where(s > 0, np.clip(50 * (s / np.nanmax(s)), 5, 50), 0.1)

    plt.close("all")
    plt.figure(figsize=(8, 4.5))
    if filt_injected is None:
        if filt_spos is None:
            plt.scatter(delta_t_over_I1, etrue_median, s=s, color="blue", marker="o")
        else:
            plt.scatter(delta_t_over_I1[~filt_spos], etrue_median[~filt_spos], s=s[~filt_spos], color="gray", marker="o")
            plt.scatter(delta_t_over_I1[filt_spos], etrue_median[filt_spos], s=s[filt_spos], facecolors="none", edgecolors="blue", marker="o")
    else:
        plt.scatter(
            delta_t_over_I1[~filt_injected], etrue_median[~filt_injected], s=s[~filt_injected], color="blue", marker="o", label="Scrambled events"
        )
        plt.scatter(
            delta_t_over_I1[filt_injected], etrue_median[filt_injected], s=s[filt_injected], color="orange", marker="o", label="Injected events"
        )

    y = np.logspace(2, 8, 201)
    x = np.array([time_delay_liv(eta, invert_H0, _y, config["evaluation"]["liv_order"]) for _y in y])

    plt.plot(np.append(-x[::-1], np.append([0], x)), np.append(y[::-1], np.append([0], y)), color="black", ls="dotted", label=rf"$\eta = {eta:3g}$")
    plt.xlabel(r"$\Delta t / I_{1}(z)$ [day]", fontsize=15)
    plt.ylabel(r"Median neutrino energy [GeV]", fontsize=15)
    plt.xscale("symlog", linthresh=min(x))
    plt.yscale("symlog", linthresh=min(y))
    plt.ylim((1e2, 1e8))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfig, dpi=300)
