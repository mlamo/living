import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import functools
import line_profiler
import numpy as np
import pandas as pd

from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d

from living.model import get_D

deltat_mjd_to_lst = 1.00273790935 * 360  # days in MJD to deg in LST
lsts_coarse = np.linspace(0, 360, 721)


class GRB:
    """Class representing a Gamma-Ray Burst (GRB).

    Attributes:
        name (str): Name/identifier of the GRB.
        time (float): Time in MJD.
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        redshift (float): Redshift of the GRB.
    """

    def __init__(self, name: str, time: float, ra: float, dec: float, redshift: float, source: str = "", fluence: float = np.nan) -> None:
        """Initializes GRB object.

        Args:
            name (str): Name/identifier of the GRB.
            time (float): Time in MJD.
            ra (float): Right ascension in degrees.
            dec (float): Declination in degrees.
            redshift (float): Redshift of the GRB.
            source (str): Source where the info for this GRB are taken from.
            fluence (float): Fluence in gamma-rays [erg cm⁻²].
        """
        self.name = name
        self.time = time
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        self.source = source
        self.fluence = fluence
        self.lst = np.nan  # local sidereal time at the GRB time computed at the detector location, in degrees
        self.minvislst = np.nan  # minimum local sidereal time for which the GRB is visible at the detector location, in degrees
        self.maxvislst = np.nan  # maximum local sidereal time for which the GRB is visible at the detector location, in degrees
        self.f_coszen = None  # function of cos(zenith) as a function of local sidereal time for the GRB at the detector location

    def __repr__(self) -> str:
        return f"{self.name}: time={self.time:.2f}, ra={self.ra} deg, dec={self.dec} deg, z={self.redshift:.2f}"

    @functools.cache
    def Dn(self, n) -> float:
        return get_D(self.redshift, n=n)

    @functools.cached_property
    def coord(self):
        return ac.SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")

    def compute_visibility(self, times: at.Time, lsts: np.ndarray, location: ac.EarthLocation):
        """Computes the upgoing visibility of the GRB at a given location.

        Args:
            times (at.Time): Array of times in MJD.
            lsts (np.ndarray): Array of local sidereal times corresponding to the times.
            location (ac.EarthLocation): Location of the detector.
        """
        coord = ac.SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")
        frame = ac.AltAz(obstime=times, location=location)
        with erfa_astrom.set(ErfaAstromInterpolator(300 * au.s)):
            c = coord.transform_to(frame)
        coszen = interp1d(lsts, np.sin(c.alt.rad), bounds_error=False, fill_value=np.sin(c.alt.rad[-1]))(lsts_coarse)
        self.f_coszen = interp1d(lsts_coarse, coszen, bounds_error=False, fill_value=(coszen[0], coszen[-1]))
        if np.all(c.alt < 0):
            self.minvislst, self.maxvislst = 0, 360
            return
        if c.alt.deg[0] > 0:
            lsts_vis = lsts[c.alt.deg < 0]
            self.minvislst, self.maxvislst = np.min(lsts_vis), np.max(lsts_vis)
        else:
            lsts_notvis = lsts[c.alt.deg > 0]
            self.maxvislst, self.minvislst = np.min(lsts_notvis), np.max(lsts_notvis)

    def is_visible(self, lst: np.ndarray) -> np.ndarray:
        """Checks if the GRB is visible at a given local sidereal time."""
        if self.minvislst is None:
            return np.ones_like(lst)
        if self.maxvislst > self.minvislst:
            return ((lst >= self.minvislst) & (lst <= self.maxvislst)).astype(float)
        return ((lst >= self.minvislst) | (lst <= self.maxvislst)).astype(float)

    def cos_zenith(self, lst: np.ndarray) -> np.ndarray:
        """Returns the cosine of the zenith angle of the GRB at a given local sidereal time."""
        if self.f_coszen is None:
            return np.ones_like(lst)
        return self.f_coszen(lst)


class GRBCatalog:
    """Class to manage a collection of GRBs for analysis.

    Attributes:
        _grbs (list): List of GRB objects.
        weighting (str): Weighting scheme for GRBs ("equal" or "distance").
    """

    def __init__(self, filename: str = "", weighting: str = "equal") -> None:
        """Initializes GRBCatalog.

        Args:
            filename (str, optional): Path to CSV file containing GRB information.
                Must have columns: mjd, ra, dec, z and name as index.
            weighting (str, optional): Weighting scheme for GRBs.
                Options: "equal" or "distance". Defaults to "equal".
        """
        self._grbs: list[GRB] = []
        if filename != "":
            self.load_csv(filename)
        self.weighting = weighting

    def __iter__(self):
        self._index = 0
        self._weights = self.compute_weights()
        return self

    def __next__(self):
        if self._index < len(self._grbs):
            grb = self._grbs[self._index]
            weight = self._weights[self._index]
            self._index += 1
            return grb, weight
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._grbs)

    def add_grb(self, grb: GRB) -> None:
        if grb.name in self.names:
            raise RuntimeWarning("The source {grb.name} is already in the catalog.")
            return
        self._grbs.append(grb)

    def load_csv(self, csvfile: str) -> None:
        """Loads GRB catalog from CSV file.

        Args:
            csvfile (str): Path to CSV file containing GRB information.
                Must have columns: mjd, ra, dec, z and name as index.
        """
        df = pd.read_csv(csvfile, index_col="name")
        for key, val in df.iterrows():
            grb = GRB(key, time=val["mjd"], ra=val["ra"], dec=val["dec"], redshift=val["z"], source=val["source"], fluence=val["fluence"])
            self.add_grb(grb)

    def filter(self, onlyGRBweb: bool = False, hasFluence: bool = False) -> None:
        """Filter GRBs with some criteria.

        Args:
            onlyGRBweb (bool, optional): If True, only keep GRBs from GRBweb. Defaults to False.
            hasFluence (bool, optional): If True, only keep GRBs with fluence information. Defaults to False.
        """
        sel_grbs = []
        for grb in self._grbs:
            selected = True
            if onlyGRBweb and grb.source != "GRBweb":
                selected = False
            if hasFluence and np.isnan(grb.fluence):
                selected = False
            if selected:
                sel_grbs.append(grb)
        self._grbs = sel_grbs

    def pick(self, num: int = 1) -> np.ndarray:
        return np.random.choice(self._grbs, size=num, p=self.compute_weights())

    @property
    def names(self) -> np.ndarray:
        return np.array([g.name for g in self._grbs])

    @property
    def times(self) -> np.ndarray:
        return np.array([g.time for g in self._grbs])

    @property
    def redshifts(self) -> np.ndarray:
        return np.array([g.redshift for g in self._grbs])

    def Dns(self, n) -> np.ndarray:
        return np.array([g.Dn(n) for g in self._grbs])

    def compute_weights(self) -> np.ndarray:
        if self.weighting == "equal":
            return 1 / len(self) * np.ones(len(self))
        if self.weighting == "Eiso":
            dists = cosmo.luminosity_distance(self.redshifts)
            w = 1 / ((1 + self.redshifts) * dists**2)
            return (w / np.sum(w)).value
        return None

    @line_profiler.profile
    def scramble(self, shuffle_time: bool = True, shuffle_ra: bool = True, shuffle_z: bool = False) -> "GRBCatalog":
        if not shuffle_time and not shuffle_ra and not shuffle_z:
            raise RuntimeWarning("Nothing is being shuffled!")
        scrambled_catalog = GRBCatalog()
        time = np.array([g.time for g in self._grbs])
        lst = np.array([g.lst for g in self._grbs])
        ra = np.array([g.ra for g in self._grbs])
        minvislst = np.array([g.minvislst for g in self._grbs])
        maxvislst = np.array([g.maxvislst for g in self._grbs])
        redshift = np.array([g.redshift for g in self._grbs])
        f_coszen = [g.f_coszen for g in self._grbs]
        if shuffle_time:
            idxs = np.arange(len(time))
            np.random.shuffle(idxs)
            time, lst = time[idxs], lst[idxs]
        if shuffle_ra:
            idxs = np.arange(len(ra))
            np.random.shuffle(idxs)
            delta_ra, ra = ra[idxs] - ra, ra[idxs]
            minvislst = (minvislst + delta_ra) % 360
            maxvislst = (maxvislst + delta_ra) % 360
            for i in range(len(f_coszen)):
                if f_coszen[i] is None:
                    continue
                f_coszen[i] = interp1d(
                    (f_coszen[i].x + delta_ra[i]) % 360, f_coszen[i].y, bounds_error=False, fill_value=(f_coszen[i].y[0], f_coszen[i].y[-1])
                )
        if shuffle_z:
            np.random.shuffle(redshift)
        for i, (grb, t, l, r, z, m1, m2) in enumerate(zip(self._grbs, time, lst, ra, redshift, minvislst, maxvislst)):
            grb = GRB(name=f"GRB-Scrambled-{i:04d}", time=t, ra=r, dec=grb.dec, redshift=z)
            grb.lst, grb.minvislst, grb.maxvislst, grb.f_coszen = l, m1, m2, f_coszen[i]
            scrambled_catalog.add_grb(grb)
        return scrambled_catalog

    def random_subset(self, N: int) -> "GRBCatalog":
        """Returns a random subset of the GRB catalog.

        Args:
            N (int): Number of GRBs to include in the subset.

        Returns:
            GRBCatalog: A new GRBCatalog instance containing the random subset of GRBs.
        """
        subset = GRBCatalog()
        idxs = np.random.choice(np.arange(len(self)), replace=False, size=N)
        for i in idxs:
            subset.add_grb(self._grbs[i])
        return subset

    def compute_visibilities(self, location: ac.EarthLocation) -> None:
        """Computes the visibility and local sidereal times of all GRBs in the catalog at a given location."""

        times = at.Time(np.linspace(56000, 56001, 86401), format="mjd", location=location)
        lsts = times.sidereal_time("apparent").deg
        isort = np.argsort(lsts)
        times, lsts = times[isort], lsts[isort]

        times_grbs = at.Time([g.time for g in self._grbs], format="mjd", location=location)
        lsts_grbs = times_grbs.sidereal_time("apparent").deg

        for grb, lst in zip(self._grbs, lsts_grbs):
            grb.lst = lst
            grb.compute_visibility(times, lsts, location)
