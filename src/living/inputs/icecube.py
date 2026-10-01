import astropy.coordinates as ac
import astropy.time as at
import glob
import histlite as hl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import logging

from io import StringIO
from sphere.distribution import fb8
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import norm, vonmises

from living.inputs.dataset import NeutrinoList, Dataset
from living.inputs.grb import GRB


def read_file(filename):

    with open(filename, "r") as file:
        content = file.read()
    # Remove '#' from the first line
    lines = content.split("\n")
    if lines[0].startswith("#"):
        lines[0] = lines[0][1:]
    modified_content = "\n".join(lines)
    # Read directly from the buffer
    return pd.read_csv(StringIO(modified_content), delimiter=r"\s+")


class ICHESENeutrinoList(NeutrinoList):

    def __init__(self, filename: str) -> None:
        d = np.load(filename)
        self._events = pd.DataFrame(data=d.tolist(), columns=d.dtype.names)
        self._events["energy"] = np.log10(self._events.energy)
        self._events["type"] = self._events.reconstruction.str.decode("utf8").str.lower()
        self._events["time"] = self._events.mjd
        self._events["lst"] = (
            at.Time(self._events["time"].to_numpy(), format="mjd", location=ac.EarthLocation.from_geodetic(lon=0, lat=-90, height=850))
            .sidereal_time("apparent")
            .deg
        )
        self._events.drop(["mjd", "reconstruction", "id", "drlogl"], axis=1, inplace=True)


class ICHESEDataset(Dataset):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplenames = ["ic"]
        self._data["ic"] = ICHESENeutrinoList("inputs/icecube/hese-12yr/data.npy")
        self._aeffs = {k: pd.read_csv(f"inputs/icecube/hese-12yr/aeff_{k}.csv") for k in ["nue", "numu", "nutau"]}
        self.range_mjd = (self._data["ic"]._events.time.min(), self._data["ic"]._events.time.max())
        self.cone_size = {"track": 10, "shower": 20}
        self._func_background_dec = None

    def psflux_to_nsig(self, spectral_index: float = 2):
        """Returns the conversion factor C = nsig / phi0 [in GeV cm²], for a per-flavour nu+anu flux dN/dE = phi0 * (E/GeV)^{-gamma}.

        Args:
            spectral_index (float, optional): Spectral index of the diffuse flux. Defaults to 2.

        Returns:
            float: Conversion factor from flux to expected number of signal events.
        """
        nsig = 0
        for _, df in self._aeffs.items():
            f = interp1d(df["Energy[GeV]"].to_numpy(), df["EffectiveArea[cm^2]"].to_numpy(), bounds_error=False, fill_value=0)
            x = np.logspace(4, 7, 1001)
            nsig += trapezoid(f(x) * np.power(x, -spectral_index), x)
        return nsig

    def compute_background_declination_pdf(self, outfig: str = None) -> None:
        """Computes background angular PDF in declination direction.

        Args:
            outfig (str, optional): Output file path for plot. If None, no plot saved.
        """
        h = hl.hist(np.sin(self._data["ic"]._events.dec.to_numpy()), bins=20, range=(-1, 1))
        hn = h.normalize()
        self._func_background_dec = hn.spline_fit()
        if outfig is not None:
            fig, ax = plt.subplots()
            hl.plot1d(ax, 1 / (2 * np.pi) * hn, crosses=True)
            x = np.linspace(-1, 1, 1001)
            ax.plot(x, self._func_background_dec(x) / (2 * np.pi))
            plt.xlabel(r"sin(dec)", fontsize=15)
            plt.ylabel(r"PDF [sr$^{-1}$]", fontsize=15)
            plt.tight_layout()
            plt.savefig(outfig)

    def evaluate_background_angular_pdf(self, evt: pd.DataFrame) -> np.ndarray:
        return self._func_background_dec(np.sin(evt.dec)) / (2 * np.pi)

    def evaluate_background_time_pdf(self, evt: pd.DataFrame) -> np.ndarray:
        return np.full(len(evt), 1 / (self.range_mjd[1] - self.range_mjd[0]))

    def evaluate_signal_angular_pdf(self, evt: pd.DataFrame, grb: GRB) -> np.ndarray:
        fb8s = [fb8(*_) for _ in evt.params]
        logger = logging.getLogger()
        logger.disabled = True
        pdfs = [f.pdf(f.spherical_coordinates_to_nu(np.pi / 2 - np.radians(grb.dec), np.radians(grb.ra))) for f in fb8s]
        logger.disabled = False
        return pdfs

    def evaluate_signal_energy_pdf(self, log10ereco: np.ndarray, log10etrue: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        sigma = np.full_like(log10ereco, 0.30)  # tracks
        sigma[evt_type == "shower"] = 0.11  # showers
        return norm.pdf(10**log10etrue, 10**log10ereco, sigma * 10**log10ereco)

    def evaluate_signal_energy_cdf(self, log10ereco: np.ndarray, log10etrue: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        sigma = np.full_like(log10ereco, 0.30)  # tracks
        sigma[evt_type == "shower"] = 0.11  # showers
        return norm.cdf(10**log10etrue, 10**log10ereco, sigma * 10**log10ereco)

    def evaluate_signal_energy_icdf(self, log10ereco: np.ndarray, q: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        sigma = np.full_like(log10ereco, 0.30)  # tracks
        sigma[evt_type == "shower"] = 0.11  # showers
        return norm.ppf(q, 10**log10ereco, sigma * 10**log10ereco)

    def compute_pdfs(self, folder_pdfs=None) -> None:
        self.compute_background_declination_pdf(os.path.join(folder_pdfs, "bkg_ang_pdf.pdf") if folder_pdfs else None)


class ICIceTracksNeutrinoList(NeutrinoList):

    def __init__(self, filenames: str) -> None:
        self._events = []
        for filename in glob.glob(filenames):
            self._events.append(read_file(filename))
        self._events = pd.concat(self._events, ignore_index=True)
        dict_renaming = {"log10(E/GeV)": "energy", "MJD[days]": "time", "RA[deg]": "ra", "Dec[deg]": "dec", "AngErr[deg]": "sigma"}
        self._events.rename(columns=dict_renaming, inplace=True)
        self._events = self._events[dict_renaming.values()]
        self._events["type"] = "track"
        self._events["ra"] = np.deg2rad(self._events.ra)
        self._events["dec"] = np.deg2rad(self._events.dec)
        self._events["lst"] = (
            at.Time(self._events["time"].to_numpy(), format="mjd", location=ac.EarthLocation.from_geodetic(lon=0, lat=-90, height=850))
            .sidereal_time("apparent")
            .deg
        )
