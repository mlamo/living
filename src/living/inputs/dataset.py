import functools
import line_profiler
import numpy as np
import pandas as pd

from abc import abstractmethod

from living.inputs.grb import GRBCatalog, GRB


class NeutrinoList:
    """Class to hold a list of neutrino candidates, either from MC or data.

    Attributes:
        _events (pd.DataFrame): DataFrame containing neutrino event information.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes NeutrinoList."""
        self._events: pd.DataFrame = None

    def __len__(self) -> int:
        return len(self._events)

    def __add__(self, other: "NeutrinoList") -> "NeutrinoList":
        d = NeutrinoList()
        d._events = pd.concat([self._events, other._events], ignore_index=True)
        return d

    @functools.cached_property
    def sum_weight_sig(self) -> float:
        return np.sum(self._events.weight_sig) if "weight_sig" in self._events.columns else None

    @line_profiler.profile
    def pick(self, num: int = 1) -> pd.DataFrame:
        """Randomly picks events weighted by signal weights.

        Args:
            num (int, optional): Number of events to pick. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing picked events.
        """
        return self._events.iloc[np.random.choice(np.arange(len(self)), size=num, p=self._events.weight_sig / self.sum_weight_sig)]

    @line_profiler.profile
    def compute_all_distances(self, grb_catalog: GRBCatalog) -> None:
        """Computes angular distances between all events and all GRBs.

        Uses vectorized computation for efficiency. Results are stored
        as columns in the events DataFrame with names like "alpha_GRBname".

        Args:
            grb_catalog (GRBCatalog): Catalog of GRBs to compute distances to.
        """

        self._events.drop(list(self._events.filter(regex="alpha")), axis=1, inplace=True)

        # Vectorized computation for all GRBs at once
        grb_coords = np.array([[grb.ra, grb.dec] for grb, _ in grb_catalog])
        event_coords = self._events[["ra", "dec"]].to_numpy()

        # Broadcast for vectorized computation
        cos_alpha = np.sin(np.deg2rad(grb_coords[:, 1]))[:, np.newaxis] * np.sin(event_coords[:, 1])[np.newaxis, :] + np.cos(
            np.deg2rad(grb_coords[:, 1])
        )[:, np.newaxis] * np.cos(event_coords[:, 1])[np.newaxis, :] * np.cos(
            np.deg2rad(grb_coords[:, 0])[:, np.newaxis] - event_coords[:, 0][np.newaxis, :]
        )

        alpha_matrix = np.rad2deg(np.arccos(np.clip(cos_alpha, -1, 1)))

        df = {}
        # Assign back to dataframe efficiently
        for i, (grb, _) in enumerate(grb_catalog):
            df[f"alpha_{grb.name}"] = alpha_matrix[i]
        self._events = pd.concat([self._events.reset_index(), pd.DataFrame(df)], axis=1)

    @line_profiler.profile
    def events(self, dataset: "Dataset", grb_catalog: GRBCatalog) -> np.recarray:

        self.compute_all_distances(grb_catalog)

        self._events.drop(list(self._events.filter(regex="sigang")), axis=1, inplace=True)
        df = {}
        for grb, _ in grb_catalog:
            df[f"sigang_{grb.name}"] = dataset.evaluate_signal_angular_pdf(self._events, grb)
        df["sigereco"] = dataset.evaluate_signal_ereco_pdf(self._events.energy.to_numpy(), self._events.type.to_numpy())
        self._events = pd.concat([self._events.reset_index(), pd.DataFrame(df)], axis=1)

        return self._events.to_records(index=False)

    @line_profiler.profile
    def get_angular_distances(self, grb: GRB) -> np.ndarray:
        """Gets angular distances to a specific GRB.

        Args:
            grb (GRB): GRB object to get distances to.

        Returns:
            np.ndarray: Angular distances in degrees.
        """
        return self._events[f"alpha_{grb.name}"].to_numpy()

    @line_profiler.profile
    def scramble(self, shuffle_time: bool = True) -> "NeutrinoList":
        """Creates a scrambled version of the neutrino list.

        Shuffles event times and updates corresponding coordinates
        to destroy any real temporal correlations.

        Args:
            shuffle_time (bool, optional): Whether to shuffle times. Defaults to True.

        Returns:
            NeutrinoList: Scrambled copy of the neutrino list.

        Raises:
            RuntimeWarning: If shuffle_time is False (no shuffling performed).
        """
        if not shuffle_time:
            raise RuntimeWarning("Nothing is being shuffled!")
        scrambled_data = NeutrinoList()
        scrambled_data._events = self._events.copy(deep=True)
        scrambled_data._events.drop(list(scrambled_data._events.filter(regex="alpha")), axis=1, inplace=True)
        scrambled_data._events.drop(list(scrambled_data._events.filter(regex="sigang")), axis=1, inplace=True)
        time = self._events.time.to_numpy()
        lst = self._events.lst.to_numpy()
        ra = self._events.ra.to_numpy()
        if shuffle_time:
            idxs = np.arange(len(time))
            np.random.shuffle(idxs)
            scrambled_data._events["time"] = time[idxs]
            scrambled_data._events["lst"] = lst[idxs]
            scrambled_data._events["ra"] = (ra + (lst[idxs] - lst)) % 360
        return scrambled_data


class Dataset:
    """Class to manage neutrino data and MC samples for analysis.

    Handles both track and shower event samples, computes PDFs,
    and provides methods for signal injection and scrambling.

    Attributes:
        samplenames (tuple): Event sample types, e.g., ("track", "shower").
        _data (dict): Dictionary of NeutrinoList objects for data.
        _mcnu (dict): Dictionary of NeutrinoList objects for MC neutrinos.
        range_mjd (list): range of Modified Julian Dates.
        cone_size (dict): size of the cones used to limit the events for which we evaluate signal PDF (in degrees)
    """

    def __init__(self, **kwargs) -> None:
        """Initializes Dataset and loads all required data files."""
        self.samplenames: list[str] = []
        self._data: dict[str, NeutrinoList] = {}
        self._mcnu: dict[str, NeutrinoList] = {}
        self.range_mjd: list[float] = [50000, 60000]
        self.cone_size: dict[float] = {}

    @property
    def data(self) -> NeutrinoList:
        """Returns combined data from all samples.

        Sums different event samples into a single NeutrinoList.

        Returns:
            NeutrinoList: Combined data from all event samples.
        """
        summed = None
        for s in self.samplenames:
            summed = self._data[s] if summed is None else summed + self._data[s]
        return summed

    """
    PDF evaluation
    """

    @abstractmethod
    def evaluate_background_angular_pdf(self, evt: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_background_time_pdf(self, evt: pd.DataFrame) -> np.ndarray:
        pass

    def evaluate_background_ereco_pdf(self, log10ereco: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        return np.ones_like(log10ereco)

    @abstractmethod
    def evaluate_signal_angular_pdf(self, evt: pd.DataFrame, grb: GRB) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_signal_energy_cdf(self, log10ereco: np.ndarray, log10etrue: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_signal_energy_icdf(self, log10ereco: np.ndarray, q: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        pass

    def evaluate_signal_ereco_pdf(self, log10ereco: np.ndarray, evt_type: np.ndarray) -> np.ndarray:
        return np.ones_like(log10ereco)

    def evaluate_signal_efficiency_pdf(self, time: np.ndarray, evt_type: np.ndarray, evt_lst: np.ndarray, grb: GRB) -> np.ndarray:
        return np.ones_like(time)

    def scramble(self, shuffle_time: bool = True) -> NeutrinoList:
        data_summed = None
        for s in self.samplenames:
            d = self._data[s].scramble(shuffle_time=shuffle_time)
            data_summed = d if data_summed is None else data_summed + d
        return data_summed

    @abstractmethod
    def inject_signal(self, grbs: GRBCatalog, eta: float, config: dict, fsig: float = None, nsig: int = None) -> NeutrinoList:
        """Prepares a fake dataset with background+signal events.

        Args:
            grbs (GRBCatalog): GRB catalog for signal generation.
            eta (float): LIV scaling parameter for time delays.
            config (dict): Dictionary containing all analysis parameters.
            fsig (float, optional): Fraction of signal events to inject.
            nsig (int, optional): Number of signal events to inject.

        Returns:
            NeutrinoList: Fake dataset with background+signal events.
        """
        pass
