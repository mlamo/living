import astropy.constants as const
import astropy.units as u
import functools
import line_profiler
import numpy as np
import scipy.integrate as si

from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from living.inputs.dataset import Dataset
    from living.inputs.grb import GRB

planck_mass = np.sqrt(const.hbar * const.c**5 / const.G).to("GeV").value
invert_H0 = (1 / cosmo.H(0)).to("day").value
Eef_NSpl_bar = 3.85e3 * 2 ** (3 / 8.0)  # Effective threshold of NSpl for eta=1 in GeV
Eth_VPE_bar = 1.85e4 * 2 ** (1 / 3.0)  # Kinematic threshold of VPE for eta=1 in GeV
invB_NSpl = 499 / 224  # Inverse of the branching ratio of the NSpl
deltat_mjd_to_lst = 1.00273790935 * 360  # days in MJD to deg in LST


def _integrand(z: float, n: int) -> float:
    """Integrand for the redshift integral computation.

    Args:
        z (float): Redshift value.
        n (int): Power of the integrand.

    Returns:
        Value of the integrand at the given redshift.
    """
    return (1 + z) ** n / np.sqrt(cosmo.Om0 * (1 + z) ** 3 + cosmo.Ode0)


def integral_redshift(redshift: float, n: int) -> float:
    """Helper function to compute a common redshift integral that
    appears in the time delay and in the probability of survival

    Args:
        redshift (float): upper limit of the integral (the lower is zero).
        n (int): Power of the integrand

    Returns:
        float: value of the integral
    """
    return si.quad(_integrand, 0, redshift, args=(n,))[0]


@functools.lru_cache(maxsize=300)
def get_D(redshift: float, n: int = 1) -> u.Quantity:
    """Returns the "distance" term appearing in the LIV time delay.
    It also produces a lookup-table for the last maxsize evaluations.

    Args:
        redshift (float): redshift of considered source.
        n (int, optional): order of the LIV effect. Defaults to 1.

    Returns:
        u.Quantity: "distance" term in time units.
    """
    return integral_redshift(redshift, n) * invert_H0


@line_profiler.profile
def time_delay_liv(eta: float, Dn: float, energy: float, n: int = 1) -> float:
    """Returns the expected absolute time delay due to LIV effects.

    Args:
        eta (float): Ratio of Planck mass and LIV scale.
        Dn (float): Dn of considered source [in days].
        energy (float): True energy of the neutrino [in GeV].
        n (int, optional): Order of the LIV correction. Defaults to 1.

    Returns:
        float: Time delay [in days].
    """
    return (n + 1) / 2 * Dn * (eta * energy / planck_mass) ** n


log10z_table = np.linspace(-5, 2, 1000)
log10J_table = np.log10(np.vectorize(integral_redshift)(10**log10z_table, 7))
log10J_inter = interp1d(log10z_table, log10J_table, kind="linear", fill_value="extrapolate")


def get_J(redshift: float) -> float:
    """Returns the redshift integral appearing in the LIV probability of survival.

    Args:
        redshift (float or float array): redshift of considered source.

    Returns:
        float: value of the redshift integral (dimensionless).
    """
    log10z = np.log10(redshift)
    log10J = log10J_inter(log10z)
    return 10**log10J


@line_profiler.profile
def survival_probability_instantaneous(energy: np.ndarray, eta: float, redshift: float, n: int = 1) -> np.ndarray:
    """Returns the survival probability (0 or 1) of a superluminal neutrino of given energy

    Args:
        energy (float array): Neutrino energy [in GeV].
        eta (float): Ratio of Planck mass and LIV scale.
        redshift (float): Redshift of considered source.
        n (int, optional): Order of the LIV correction. Defaults to 1.

    Returns:
        np.ndarray: Survival probability (0 or 1)
    """
    if n != 1:
        raise RuntimeError(f"Order n={n} is not yet implemented")

    if eta == 0:
        return np.ones_like(energy)

    energy_max_liv = Eef_NSpl_bar * np.power(eta, -3 / 8.0) / (1 + redshift)
    return (energy < energy_max_liv).astype(float)


@line_profiler.profile
def survival_probability_integral(energy: np.ndarray, eta: float, redshift: float, n: int = 1) -> np.ndarray:
    """Returns the survival probability of a superluminal neutrino of given energy, using the full expression.

    Args:
        energy (float array): Neutrino energy [in GeV].
        eta (float): Ratio of Planck mass and LIV scale.
        redshift (float or float array): Redshift of considered source.
        n (int, optional): Order of the LIV correction. Defaults to 1.

    Returns:
        np.ndarray: Survival probability
    """
    if n != 1:
        raise RuntimeError(f"Order n={n} is not yet implemented")

    if eta == 0:
        return np.ones_like(energy)

    # Compute energy part of the optical depth
    tau_ene = (energy / Eef_NSpl_bar) ** 8 * np.power(eta, 3)

    # Compute redshift part of the optical depth
    J_zsrc = get_J(redshift)
    Eth_VPE = Eth_VPE_bar * np.power(eta, -1 / 3)
    zstar = np.clip((Eth_VPE / energy) - 1, 1e-5, redshift)
    J_zstar = get_J(zstar)

    # Return exponential of minus the optical depth
    tau = tau_ene * (invB_NSpl * J_zsrc - (invB_NSpl - 1) * J_zstar)
    return np.exp(-tau)


def survival_probability(energy: np.ndarray, eta: float, redshift: float, n: int = 1, method: str = "integral") -> np.ndarray:
    """Compute the survival probability of a superluminal neutrino.

    Args:
        energy: Neutrino energy [in GeV].
        eta: Ratio of Planck mass and LIV scale.
        redshift: Redshift of considered source.
        n (int, optional): Order of the LIV correction. Defaults to 1.
        method: Method to use for computation. Options: "instantaneous", "integral". Defaults to "integral".

    Returns:
        Survival probability values.
    """
    if method == "instantaneous":
        return survival_probability_instantaneous(energy, eta, redshift, n)
    elif method == "integral":
        return survival_probability_integral(energy, eta, redshift, n)
    else:
        raise RuntimeError("Unknown method for `survival_probability`.")


@line_profiler.profile
def time_pdf(eta: float, evt_times, evt_log10erecos, evt_types, evt_lsts, dataset: "Dataset", grb: "GRB", config: dict) -> np.ndarray:
    """Time PDF that integrates to 1 over delta_t for fixed energy, eta, grb, and intrinsic time delay (τ₋, τ₊).
    It is computed as dP/dt = C / [(1+z) * (τ₊ - τ₋)] * f_live(t_nu)
                                * [ 1/2 *int_{E₊}^{E₋} [dP/dE dE]
                                  + 1/2 *int_{-E₋}^{-E₊} [dP/dE * f_surv(E) dE] ] where:
        - dP/dE is the energy PDF of the true energy for the measured reconstructed energy.
        - E₊ = β * (∆t - (1+z)*τ₊) and E₋ = β * (∆t - (1+z)*τ₋)
        - f_surv is the survival probability for superluminal neutrinos because of decay.
        - f_live is the event selection efficiency at the time of the neutrino.
        - C is a normalisation constant to account for the presence of f_surv and f_live terms.

    Args:
        eta (float): LIV scaling parameter.
        evt_times (np.array): Event times.
        evt_log10erecos (np.array): Event reconstructed log10(energy).
        evt_types (np.array): Event types.
        evt_lsts (np.array): Event local sidereal times.
        dataset (Dataset): Dataset object.
        grb (GRB): GRB object containing time and redshift information.
        config (dict): Dictionary containing all analysis parameters.

    Returns:
        np.ndarray: Time PDF values for each event.
    """
    n_liv = config["evaluation"]["liv_order"]
    tau_m = config["grbs"]["intrinsic_time_delay"][0] / 86400
    tau_p = config["grbs"]["intrinsic_time_delay"][1] / 86400

    include_sub = config["evaluation"]["components"] != "superluminal"
    include_sup = (config["evaluation"]["components"] != "subluminal") and (
        (n_liv % 2 != 0) or (config["evaluation"]["components"] == "superluminal")
    )
    include_decay = config["evaluation"]["include_decay"]
    include_eff = config["evaluation"]["include_neutrino_efficiency"]

    delta_t = evt_times - grb.time

    # Special case: no LIV effects - uniform distribution over intrinsic window
    if eta == 0:
        if (tau_m == tau_p == 0) or config["evaluation"]["approximate_time_pdf"]:
            return 0
        return ((delta_t >= tau_m) & (delta_t <= tau_p)) / (tau_p - tau_m)

    sig_ene = dataset.evaluate_signal_energy_pdf
    sig_enecum = dataset.evaluate_signal_energy_cdf
    sig_eff = dataset.evaluate_signal_efficiency_pdf

    beta = 1 / time_delay_liv(eta, grb.Dn(n_liv), 1, n_liv)

    if config["evaluation"]["approximate_time_pdf"]:
        norm = 0
        pdf = np.zeros_like(delta_t)
        if include_sub:
            filt_sub = delta_t > tau_p
            log10etrue = 1 / n_liv * np.log10(beta * delta_t[filt_sub])
            pdf[filt_sub] = (
                beta / (n_liv * np.power(10, log10etrue * (n_liv - 1))) * sig_ene(evt_log10erecos[filt_sub], log10etrue, evt_types[filt_sub])
            )
            norm += 1
        if include_sup:
            filt_sup = delta_t < tau_m
            log10etrue = 1 / n_liv * np.log10(-beta * delta_t[filt_sup])
            pdf[filt_sup] = (
                beta / (n_liv * np.power(10, log10etrue * (n_liv - 1))) * sig_ene(evt_log10erecos[filt_sup], log10etrue, evt_types[filt_sup])
            )
            if include_decay:
                pdf[filt_sup] *= survival_probability(
                    -beta * delta_t[filt_sup], eta, grb.redshift, n_liv, config["model"]["survival_probability_method"]
                )
                norm += sig_enecum(
                    evt_log10erecos, np.full_like(evt_log10erecos, np.log10(Eef_NSpl_bar * np.power(eta, -3 / 8.0) / (1 + grb.redshift))), evt_types
                )
            else:
                norm += 1
        pdf /= norm
        return pdf

    if n_liv > 1 and include_sup:
        raise RuntimeError(f"Order n={n_liv} is not yet implemented for superluminal component.")

    z_plus_1 = 1 + grb.redshift
    norm = 1 / (z_plus_1 * (tau_p - tau_m))

    # Vectorized energy bounds computation
    beta_delta_t_m = beta * (delta_t - z_plus_1 * tau_m)
    beta_delta_t_p = beta * (delta_t - z_plus_1 * tau_p)

    # Compute subluminal term
    pdf_sub = np.zeros_like(delta_t)
    if include_sub:
        filt = delta_t > z_plus_1 * tau_m
        energy_m_clipped = np.clip(beta_delta_t_m[filt], 1, np.inf) ** (1 / n_liv)
        energy_p_clipped = np.clip(beta_delta_t_p[filt], 1, np.inf) ** (1 / n_liv)
        pdf_sub[filt] = sig_enecum(evt_log10erecos[filt], np.log10(energy_m_clipped), evt_types[filt]) - sig_enecum(
            evt_log10erecos[filt], np.log10(energy_p_clipped), evt_types[filt]
        )
        pdf_sub[pdf_sub < 0] = 0

    # Compute superluminal term
    pdf_sup = np.zeros_like(delta_t)
    if include_sup:
        filt = delta_t < z_plus_1 * tau_p
        neg_energy_p_clipped = np.clip(-beta_delta_t_p[filt], 1, np.inf) ** (1 / n_liv)
        neg_energy_m_clipped = np.clip(-beta_delta_t_m[filt], 1, np.inf) ** (1 / n_liv)
        if include_decay:
            npts = 50
            x = np.linspace(0, 1, npts)
            log10etrue = (
                np.log10(neg_energy_m_clipped[:, np.newaxis])
                + x[np.newaxis, :] * (np.log10(neg_energy_p_clipped) - np.log10(neg_energy_m_clipped))[:, np.newaxis]
            )
            log10etruec = 0.5 * (log10etrue[:, 1:] + log10etrue[:, :-1])
            log10ereco = np.repeat(evt_log10erecos[filt], npts)  # shape = (nevts*npts,)
            names = np.repeat(evt_types[filt], npts)  # shape = (nevts*npts,)
            cdfs = sig_enecum(log10ereco, log10etrue.flatten(), names).reshape(log10etrue.shape)
            pdf_sup[filt] = np.sum(
                np.diff(cdfs, axis=1)
                * survival_probability(10**log10etruec, eta, grb.redshift, n_liv, config["model"]["survival_probability_method"]),
                axis=1,
            )
        else:
            pdf_sup[filt] = sig_enecum(evt_log10erecos[filt], np.log10(neg_energy_p_clipped), evt_types[filt]) - sig_enecum(
                evt_log10erecos[filt], np.log10(neg_energy_m_clipped), evt_types[filt]
            )
        pdf_sup[pdf_sup < 0] = 0

    pdf = norm * (pdf_sub + pdf_sup)
    if include_sub and include_sup:
        pdf /= 2
    evt_sigeff = np.ones_like(pdf_sub)
    if include_eff:
        evt_sigeff = sig_eff(evt_times, evt_types, evt_lsts, grb)
        pdf *= evt_sigeff

    if np.all(pdf == 0):
        return pdf

    # Monte Carlo integration of the time PDF to renormalise it
    if include_decay or include_eff:
        ntoys = 100
        q = np.random.random((np.count_nonzero(pdf > 0), ntoys))
        e_flat = np.repeat(evt_log10erecos[pdf > 0], ntoys)
        q_flat = q.flatten()
        n_flat = np.repeat(evt_types[pdf > 0], ntoys)
        etrue = dataset.evaluate_signal_energy_icdf(e_flat, q_flat, n_flat)
        eff_sub = np.zeros_like(q)
        eff_sup = np.zeros_like(q)

        # Renormalisation needed if decay is included
        psurv = np.ones_like(q)
        if include_sup and include_decay:
            psurv = survival_probability(etrue, eta, grb.redshift, n_liv, config["model"]["survival_probability_method"]).reshape(q.shape)

        # Renormalisation needed if efficiency is included
        if include_eff:
            delta_t = np.power(etrue, n_liv) / beta
            if include_sub:
                eff_sub = sig_eff(grb.time + delta_t, n_flat, (grb.lst + deltat_mjd_to_lst * delta_t) % 360, grb).reshape(q.shape)
            if include_sup:
                _eff_sup = np.zeros_like(q_flat)
                _eff_sup[psurv.flatten() > 0] = sig_eff(
                    grb.time - delta_t[psurv.flatten() > 0],
                    n_flat[psurv.flatten() > 0],
                    (grb.lst - deltat_mjd_to_lst * delta_t[psurv.flatten() > 0]) % 360,
                    grb,
                )
                eff_sup = _eff_sup.reshape(q.shape)
        else:
            if include_sub:
                eff_sub = np.ones_like(q)
            if include_sup:
                eff_sup = np.ones_like(q)
        if include_sub and include_sup:
            eff_sub /= 2
            eff_sup /= 2

        # Compute integral
        integral = np.nanmean(eff_sub, axis=1) + np.nanmean(psurv * eff_sup, axis=1)
        norm = np.full_like(integral, 0)
        norm[integral > 0] = np.clip(1 / integral[integral > 0], 0.5, 10 / evt_sigeff[pdf > 0][integral > 0])

        # Apply normalization corrections
        pdf[pdf > 0] *= norm

    return pdf
