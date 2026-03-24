"""
SensorDataset abstraction for handling sensor data in urban water systems.

This module defines the 'SensorDataset' class used to handle time-series datasets collected by online sensors in the OpenWaterLab (OWL) ecosystem.

The SensorDataset class extends the Dataset class by providing additional functionalities for gap filling using various algorithms.

Classes
-------
SensorDataset
    Extends Dataset class for time-series sensor data measurements.
"""

from __future__ import annotations



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings as wn
import random as rn
import inspect

from .dataset import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, Tuple, Literal, Any, Hashable, Iterable, List, Callable
import logging
import types
import inspect
logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """
    Dataset class for handling online sensor time-series.

    This class extends the base ``Dataset`` with functionality tailored to
    continuous or high-frequency sensor measurements commonly encountered in
    water and wastewater systems. In particular it adds various gap filling algorithms
    to the base class.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-indexed sensor data, where rows correspond to timestamps and
        columns correspond to sensor signals.
    timedata_column : str, default="index"
        Column used as the time reference, or ``"index"`` if the index
        represents time.
    data_type : str, default="WWTP"
        Type of data represented in the dataset.
    experiment_tag : str, default="No tag given"
        Label identifying the dataset or experiment.
    time_unit : str, optional
        Unit of the time axis if it is numeric rather than datetime-like
        (e.g., ``"min"``, ``"h"``, ``"d"``).

    Attributes
    ----------
    Inherits all attributes from 'Dataset'.

    Notes
    -----
    - The class is intended for continuous or regularly sampled sensor data.
    - It inherits all core functionality from ``Dataset`` and may extend it
      with sensor-specific validation, filtering, gap filling and resampling utilities.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        timedata_column: str = "index",
        data_type: str = "WWTP",
        experiment_tag: str = "No tag given",
        time_unit: Optional[str] = None,
    ) -> None:
        super().__init__(
            data=data,
            timedata_column=timedata_column,
            data_type=data_type,
            experiment_tag=experiment_tag,
            time_unit=time_unit,
        )

        self.filled: pd.DataFrame = pd.DataFrame(index=self.data.index, dtype=float)

        base_meta = getattr(self, "meta_valid", None)
        if base_meta is not None and not base_meta.empty:
            self.meta_filled: pd.DataFrame = base_meta.reindex(self.data.index)
        else:
            self.meta_filled = pd.DataFrame(index=self.data.index)

        self.filling_error: pd.DataFrame = pd.DataFrame(
            data=np.nan,
            index=pd.Index(self.data.columns, name="signal"),
            columns=["imputation error [%]"],
        )

        self._filling_warning_issued: bool = False
        self._rain_warning_issued: bool = False



    def drop_index_duplicates(
        self,
        *,
        keep: Literal["first", "last", False] = "first",
        sort_index: bool = True,
        print_number: bool = True
    ) -> None:
        """
        Remove duplicate index entries from the dataset.

        The method drops duplicated index rows from the data and applies the same
        selection to associated metadata and auxiliary data structures.

        Parameters
        ----------
        keep : {"first", "last", False}, default="first"
            Which duplicate entries to retain. If ``False``, all occurrences of
            duplicated index labels are removed.
        sort_index : bool, default=True
            If ``True``, sort all affected data structures by index after removal.
        print_number : bool, default=True
            If ``True``, print the number of duplicate rows removed.

        Returns
        -------
        None

        Notes
        -----
        - The same positional mask is applied consistently across ``data`` and
        related attributes (e.g., metadata and filled data).
        - The operation modifies the dataset in place.
        """
        if self.data is None or self.data.empty:
            return None

        if keep is False:
            counts = self.data.index.value_counts()
            n_dupes = int(counts[counts > 1].sum())
        else:
            n_total = int(len(self.data))
            n_unique = int(self.data.index.nunique())
            n_dupes = n_total - n_unique

        orig_index = self.data.index
        keep_mask = ~orig_index.duplicated(keep=keep) if keep in ("first", "last") else ~orig_index.duplicated(keep=False)
        keep_mask_pos = np.asarray(keep_mask, dtype=bool)

        self.data = self.data.iloc[keep_mask_pos]

        for attr in ("meta_valid", "meta_filled", "filled"):
            frame = getattr(self, attr, None)
            if isinstance(frame, pd.DataFrame):
                if len(frame) == len(orig_index):
                    setattr(self, attr, frame.iloc[keep_mask_pos])
                else:
                    
                    setattr(self, attr, frame.reindex(self.data.index))
            else:
                setattr(self, attr, pd.DataFrame(index=self.data.index))

        if sort_index:
            self.data = self.data.sort_index()
            self.meta_valid = self.meta_valid.sort_index()
            self.meta_filled = self.meta_filled.sort_index()
            self.filled = self.filled.sort_index()

        self._update_time()

        if len(self.data.index) >= 2 and self.data.index.dtype == object:
            wn.warn(
                "Index has object dtype; ordering may be unexpected. Consider converting "
                "to datetime or numeric and calling .sort_index().",
                RuntimeWarning,
                stacklevel=2,
            )
        if print_number:
            print(f"{n_dupes} rows have been dropped as duplicates.")
        return None
    

    def calc_total_proportional(
        self,
        Q_tot: str,
        Q: Sequence[str],
        conc: Sequence[str],
        *,
        new_name: str = "new",
        unit: str = "mg/l",
        filled: bool = False,
    ) -> None:
        """
        Compute a flow-weighted total concentration.

        The method calculates a combined concentration based on multiple
        contributing flows and their associated concentrations.

        Parameters
        ----------
        Q_tot : str
            Name of the column containing the total flow (denominator).
        Q : sequence of str
            Names of the columns representing contributing flows.
        conc : sequence of str
            Names of the columns representing concentrations corresponding
            to each flow in ``Q`` (same order).
        new_name : str, default="new"
            Name of the output column to create.
        unit : str, default="mg/l"
            Unit string assigned to the new column.
        filled : bool, default=False
            If ``True``, perform the calculation using filled data.
            Otherwise, use the original dataset.

        Returns
        -------
        None

        Notes
        -----
        - The total concentration is computed as:
        ``sum(Q_i * conc_i) / Q_tot``.
        - The result is added as a new column to the selected dataset.
        - The dataset is modified in place.
        """
        df = self.filled if filled else self.data
        if df is None or df.empty:
            raise ValueError("No data available to compute proportional total.")

        if Q_tot not in df.columns:
            raise KeyError(f"Total flow column '{Q_tot}' not found.")
        if not Q or not conc:
            raise ValueError("Q and conc must be non-empty sequences of column names.")
        if len(Q) != len(conc):
            raise ValueError(f"Q and conc must have the same length (got {len(Q)} vs {len(conc)}).")

        missing_q = [c for c in Q if c not in df.columns]
        missing_c = [c for c in conc if c not in df.columns]
        if missing_q or missing_c:
            missing = ", ".join(missing_q + missing_c)
            raise KeyError(f"Missing columns: {missing}")

        Q_tot_s = pd.to_numeric(df[Q_tot], errors="coerce")

        numerator = pd.Series(0.0, index=df.index)
        for Qi, Ci in zip(Q, conc):
            Qi_s = pd.to_numeric(df[Qi], errors="coerce")
            Ci_s = pd.to_numeric(df[Ci], errors="coerce")
            numerator = numerator.add(Qi_s * Ci_s, fill_value=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = numerator.divide(Q_tot_s)
        result = result.replace([np.inf, -np.inf], np.nan)

        df[new_name] = result.astype(float)

        if not filled:
            self.data = df
            if hasattr(self, "columns"):
                try:
                    self.columns = np.array(self.data.columns)
                except Exception:
                    pass

        try:
            if hasattr(self, "units") and isinstance(self.units, dict):
                self.units[new_name] = unit
            elif hasattr(self, "units") and isinstance(self.units, pd.DataFrame):
                if new_name in self.units.columns:
                    self.units.loc[:, new_name] = unit
                else:
                    try:
                        self.units[new_name] = unit
                    except Exception:
                        wn.warn(
                            "Could not set unit on units DataFrame; please update manually.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            else:
                pass
        except Exception:
            wn.warn(
                "Something went wrong while updating units; verify self.units.",
                RuntimeWarning,
                stacklevel=2,
            )

        return None


    def calc_daily_average(
        self,
        column_name: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        plot: bool = False,
    ) -> Optional[Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]]:
        """
        Compute daily averages and standard deviations for a column.

        The method aggregates data on a daily basis and stores the results
        internally for further analysis or visualization.

        Parameters
        ----------
        column_name : str
            Name of the column to aggregate.
        arange : tuple, optional
            Range of data over which the aggregation is performed.
            Must be compatible with the index type. If ``None``, the full
            dataset is used.
        plot : bool, default=False
            If ``True``, generate a plot showing daily mean values with
            standard deviation as error bars.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
            If ``plot=True``, returns the generated figure and axes.
            Otherwise, returns ``None``.

        Notes
        -----
        - If the index is a ``DatetimeIndex``, data is resampled by calendar day.
        - If the index is numeric, days are defined as integer bins ``[n, n+1)``.
        - Results are stored in ``self.daily_average[column_name]`` as a DataFrame
        with columns ``["day", "mean", "std"]``.
        - The dataset itself is not modified.
        """
        if not hasattr(self, "daily_average") or not isinstance(getattr(self, "daily_average"), dict):
            self.daily_average = {}

        if column_name not in self.data.columns:
            raise KeyError(f"Column '{column_name}' not found in data.")

        try:
            if arange is None:
                series = self.data[column_name].copy()
            else:
                series = self.data.loc[arange[0]:arange[1], column_name].copy()
        except TypeError as e:
            raise TypeError(
                f"Slicing not possible for index type {type(self.data.index[0])} "
                f"with arange element types {type(arange[0])}, {type(arange[1])}. "
                "Ensure arange values are compatible with the index."
            ) from e

        if series.empty:
            wn.warn("Selected range is empty; no daily averages computed.", RuntimeWarning, stacklevel=2)
            self.daily_average[column_name] = pd.DataFrame(columns=["day", "mean", "std"])
            return None

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            wn.warn("No numeric samples in selected range; no daily averages computed.", RuntimeWarning, stacklevel=2)
            self.daily_average[column_name] = pd.DataFrame(columns=["day", "mean", "std"])
            return None

        if isinstance(series.index, pd.DatetimeIndex):
            daily_mean = series.resample("D").mean()
            daily_std = series.resample("D").std()
            to_return = pd.DataFrame({
                "day": daily_mean.index,
                "mean": daily_mean.values,
                "std": daily_std.values,
            })

        elif np.issubdtype(series.index.dtype, np.number):
            days = np.floor(series.index.values).astype(int)
            df_tmp = pd.DataFrame({"day": days, "val": series.values})
            grouped = df_tmp.groupby("day")
            to_return = pd.DataFrame({
                "day": grouped["day"].first().index,
                "mean": grouped["val"].mean().values,
                "std": grouped["val"].std().values,
            })
        else:
            raise TypeError(
                "Unsupported index type. Use a DatetimeIndex or numeric index for daily averaging."
            )

        self.daily_average[column_name] = to_return

        if plot:
            fig, ax = plt.subplots(figsize=(16, 6))
            if isinstance(series.index, pd.DatetimeIndex):
                ax.errorbar(pd.to_datetime(to_return["day"]), to_return["mean"], yerr=to_return["std"], fmt="o")
                ax.set_xlabel("Time")
            else:
                ax.errorbar(to_return["day"], to_return["mean"], yerr=to_return["std"], fmt="o")
                ax.set_xlabel("Day (index units)")
            ax.set_ylabel(column_name)
            ax.tick_params(labelsize=12)
            fig.tight_layout()
            return fig, ax

        return None

#     ###############################################################################
#     ##                        FILLING HELP FUNCTIONS                             ##
#     ###############################################################################
    
    def _reset_meta_filled(self, data_name: Optional[str] = None) -> None:
        """
        Reset the `meta_filled` DataFrame.

        Parameters
        ----------
        data_name : str, optional
            If provided, only reset the given column in `meta_filled` to match
            `meta_valid`. If None, reset the entire DataFrame.
        
        Returns
        -------
        None

        Notes
        -----
        - Ensures index alignment with `self.data.index`.
        - If `data_name` does not exist in `meta_valid`, a warning is issued.
        """
        if data_name is None:
            if hasattr(self, "meta_valid") and isinstance(self.meta_valid, pd.DataFrame):
                self.meta_filled = self.meta_valid.copy().reindex(self.data.index)
            else:
                self.meta_filled = pd.DataFrame(index=self.data.index)
        else:
            if hasattr(self, "meta_valid") and data_name in self.meta_valid.columns:
                if not hasattr(self, "meta_filled") or not isinstance(self.meta_filled, pd.DataFrame):
                    self.meta_filled = pd.DataFrame(index=self.data.index)
                if data_name not in self.meta_filled.columns:
                    self.meta_filled[data_name] = None
                self.meta_filled[data_name] = self.meta_valid[data_name].copy()
            else:
                wn.warn(
                    f"Column '{data_name}' not found in meta_valid; nothing was reset.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    
    def add_to_filled(self, column_names: Union[str, Sequence[str]]) -> None:
        """
        Add one or more columns into `self.filled`, seeding them with the *validated*
        (i.e., 'original' in meta_valid) values from `self.data`. All other rows
        become NaN (to be filled later by imputation methods).

        Parameters
        ----------
        column_names : str or sequence of str
            Column(s) to add/seed into `self.filled`.
        
        Returns
        -------
        None

        Notes
        -----
        - If `meta_valid` is missing or does not contain a column tag, the entire
        column is copied (with a warning).
        - `self.filled` is created/reindexed to match `self.data.index`.
        """
        self._plot = "filled"

        if isinstance(column_names, str):
            names = [column_names]
        else:
            names = list(column_names)

        if not hasattr(self, "filled") or not isinstance(self.filled, pd.DataFrame):
            self.filled = pd.DataFrame(index=self.data.index)
        else:
            self.filled = self.filled.reindex(self.data.index)

        meta = getattr(self, "meta_valid", None)
        has_meta = isinstance(meta, pd.DataFrame) and not meta.empty

        for col in names:
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

            series = self.data[col].copy()

            if has_meta and (col in meta.columns):
                mask_original = (meta[col].reindex(self.data.index) == "original")
                seeded = series.where(mask_original, np.nan)
            else:
                wn.warn(
                    f"meta_valid missing or no tags for '{col}'; copying entire column.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                seeded = series

            self.filled[col] = pd.to_numeric(seeded, errors="coerce")
            self.filled = self.filled.reindex(self.data.index)

        return None
                
    
    def _add_to_meta(self, to_fill: str) -> None:
        """
        Ensure a column is consistently represented across metadata and filled data.

        Parameters
        ----------
        to_fill : str
            Name of the column to initialize and align across metadata and
            auxiliary data structures.

        Returns
        -------
        None

        Notes
        -----
        - Ensures that ``meta_valid``, ``meta_filled``, and ``filled`` exist and are
        aligned with ``self.data.index``.
        - Creates missing structures and removes duplicate indices where necessary.
        - Initializes metadata tags:
        - ``meta_valid[to_fill]`` defaults to ``"original"`` where missing.
        - ``meta_filled[to_fill]`` is derived from ``meta_valid`` if not present.
        - Initializes ``filled[to_fill]`` using validated/original values from
        ``self.data[to_fill]``; non-validated entries are set to NaN.
        - The operation modifies the dataset in place.
        """
        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        idx = self.data.index

        if not hasattr(self, "meta_valid") or not isinstance(self.meta_valid, pd.DataFrame):
            self.meta_valid = pd.DataFrame(index=idx)
        else:
            if not self.meta_valid.index.is_unique:
                self.meta_valid = self.meta_valid.loc[~self.meta_valid.index.duplicated(keep="first")]
            self.meta_valid = self.meta_valid.reindex(idx)

        if to_fill not in self.meta_valid.columns:
            self.meta_valid[to_fill] = "original"
        else:
            self.meta_valid[to_fill] = (
                self.meta_valid[to_fill]
                .astype(object)
                .fillna("original")
                .replace({"!!": "original"})
            )

        if not hasattr(self, "meta_filled") or not isinstance(self.meta_filled, pd.DataFrame):
            self.meta_filled = pd.DataFrame(index=idx)
        else:
            if not self.meta_filled.index.is_unique:
                self.meta_filled = self.meta_filled.loc[~self.meta_filled.index.duplicated(keep="first")]
            self.meta_filled = self.meta_filled.reindex(idx)

        if to_fill not in self.meta_filled.columns:
            self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()
        else:
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .astype(object)
                .where(self.meta_filled[to_fill].notna(), self.meta_valid[to_fill])
                .fillna("original")
                .replace({"!!": "original"})
            )

        if not hasattr(self, "filled") or not isinstance(self.filled, pd.DataFrame):
            self.filled = pd.DataFrame(index=idx)
        else:
            if not self.filled.index.is_unique:
                self.filled = self.filled.loc[~self.filled.index.duplicated(keep="first")]
            self.filled = self.filled.reindex(idx)

        if to_fill not in self.filled.columns:
            mask_original = (self.meta_valid[to_fill] == "original")
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce").where(mask_original)
        else:
            self.filled[to_fill] = pd.to_numeric(self.filled[to_fill], errors="coerce")

        self.meta_valid[to_fill] = self.meta_valid[to_fill].replace({"!!": "original"}).fillna("original")
        self.meta_filled[to_fill] = self.meta_filled[to_fill].replace({"!!": "original"}).fillna("original")

        return None
    

    def _warning(
        self,
        message,                 
        category,                
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ) -> None:
        """
        Format and display warnings using a custom style.

        Parameters
        ----------
        message : str
            Warning message.
        category : type
            Warning category.
        filename : str
            Name of the source file where the warning occurred.
        lineno : int
            Line number where the warning occurred.
        file : file-like object, optional
            Output stream for the warning.
        line : str, optional
            Source code line associated with the warning.

        Returns
        -------
        None

        Notes
        -----
        - Intended as a custom formatter for ``warnings.showwarning``.
        - Produces messages in the form:
        ``"<filename>:<lineno>: <CategoryName>: <message>"``.
        """
        cat_name = category.__name__ if hasattr(category, "__name__") else str(category)
        msg_text = str(message)
        out = f"{filename}:{lineno}: {cat_name}: {msg_text}"
        stream = file if file is not None else wn._showwarnmsg_impl.__self__ if hasattr(wn._showwarnmsg_impl, "__self__") else None
        if stream and hasattr(stream, "write"):
            try:
                stream.write(out + "\n")
            except Exception:
                print(out)  
        else:
            print(out)


    def _use_custom_warning_format(self, enable: bool = True) -> None:
        """
        Enable or disable custom warning formatting.

        Parameters
        ----------
        enable : bool, default=True
            If ``True``, activate the custom warning formatter. If ``False``,
            restore the default warning behavior.

        Returns
        -------
        None

        Notes
        -----
        - Modifies the global ``warnings.showwarning`` hook.
        - Affects all warnings in the current Python process.
        """
        if enable:
            wn.showwarning = types.MethodType(self._warning, self)  
        else:
            wn.showwarning = wn._showwarning_orig if hasattr(wn, "_showwarning_orig") else wn._showwarning  


    def _filling_warning(self, *, use_custom_format: bool = False, stacklevel: int = 2) -> None:
        """
        Issue a one-time warning related to filling workflows.

        Parameters
        ----------
        use_custom_format : bool, default=False
            If ``True``, use the custom warning formatter.
        stacklevel : int, default=2
            Stack level passed to the warning to control the reported location.

        Returns
        -------
        None

        Notes
        -----
        - The warning is displayed only on the first invocation.
        - Intended to inform users about filling-related behavior or assumptions.
        """
        if getattr(self, "_filling_warning_issued", False):
            return

        if use_custom_format:
            try:
                if not hasattr(wn, "_showwarning_orig"):
                    wn._showwarning_orig = wn.showwarning
                self._use_custom_warning_format(True)
            except Exception:
                pass  

        wn.warn(
            "When using filling functions, start with small gaps and progressively "
            "move to larger gaps. This improves algorithm reliability. "
            "This notice is shown only once.",
            UserWarning,
            stacklevel=stacklevel,
        )

        if use_custom_format:
            try:
                self._use_custom_warning_format(False)
            except Exception:
                pass

        self._filling_warning_issued = True


    def _rain_warning(self, *, use_custom_format: bool = False, stacklevel: int = 2) -> None:
        """
        Issue a warning for operations affecting rain or high-event data.

        Parameters
        ----------
        use_custom_format : bool, default=False
            If ``True``, use the custom warning formatter.
        stacklevel : int, default=2
            Stack level passed to the warning to control the reported location.

        Returns
        -------
        None

        Notes
        -----
        - The warning is displayed on every invocation.
        - Intended to highlight potential impacts on rain or high-flow event data.
        """
        if use_custom_format:
            try:
                if not hasattr(wn, "_showwarning_orig"):
                    wn._showwarning_orig = wn.showwarning
                self._use_custom_warning_format(True)
            except Exception:
                pass

        wn.warn(
            "Data points obtained during rain/high events will be replaced. "
            "Ensure the chosen method is appropriate for gaps during rain events.",
            UserWarning,
            stacklevel=stacklevel,
        )

        if use_custom_format:
            try:
                self._use_custom_warning_format(False)
            except Exception:
                pass
    
    def _check_rain(
        self,
        arange: Optional[Tuple[object, object]] = None,
    ) -> bool:
        """
        Check whether a data range overlaps with rain or high-event periods.

        Parameters
        ----------
        arange : tuple, optional
            Range of data to check. If ``None``, the full dataset range is used.

        Returns
        -------
        bool
            ``True`` if a rain or high-event period is detected (and a warning is issued),
            otherwise ``False``.

        Notes
        -----
        - Intended to identify overlaps between the selected data range and
        rain or high-flow events.
        - A warning is issued when such an overlap is detected.
        """
        if getattr(self, "data_type", None) != "WWTP":
            return False

        highs = getattr(self, "highs", None)
        if not isinstance(highs, pd.DataFrame) or "highs" not in highs.columns or highs.empty:
            return False

        if arange is None:
            sub = highs["highs"]
        else:
            try:
                sub = highs.loc[arange[0]:arange[1], "highs"]
            except Exception:
                return False

        if sub.sum() > 0:
            self._rain_warning()
            return True
        return False
    
    def _check_daily_profile(
        self,
        column_name: Optional[str] = None,
        *,
        return_df: bool = False,
    ) -> Union[bool, pd.DataFrame]:
        """
        Check for the existence of a daily profile.

        Parameters
        ----------
        column_name : str, optional
            If provided, check for a profile associated with this column.
        return_df : bool, default=False
            If ``True``, return the profile DataFrame instead of a boolean.

        Returns
        -------
        bool or pandas.DataFrame
            If ``return_df=False``, returns ``True`` if the profile exists,
            otherwise ``False``. If ``return_df=True``, returns the requested
            profile DataFrame.

        Raises
        ------
        AttributeError
            If ``self.daily_profile`` is not defined.
        TypeError
            If ``self.daily_profile`` is not dict-like.
        KeyError
            If ``column_name`` is provided but not found.

        Notes
        -----
        - Profiles are expected to be stored in ``self.daily_profile``.
        - The method does not modify the dataset.
        """
        if not hasattr(self, "daily_profile"):
            raise AttributeError(
                "self.daily_profile doesn't exist yet. Run calc_daily_profile(...) first."
            )
        if not isinstance(self.daily_profile, dict):
            raise TypeError(
                "self.daily_profile should be a dict. Recompute via calc_daily_profile(...)."
            )

        if column_name is None:
            if not self.daily_profile:
                return False if not return_df else pd.DataFrame()
            if return_df:
                return pd.concat(self.daily_profile, names=["column", "time_of_day"])
            return True

        if column_name not in self.daily_profile:
            return False if not return_df else pd.DataFrame()

        return self.daily_profile[column_name] if return_df else True
    
    def _align_aux_frames(self, keep: str = "first") -> None:
        """
        Align auxiliary data structures with the main dataset index.

        Parameters
        ----------
        keep : {"first", "last"}, default="first"
            Which duplicate index entries to retain when enforcing uniqueness.

        Returns
        -------
        None

        Notes
        -----
        - Ensures auxiliary frames have unique indices.
        - Aligns all auxiliary frames to ``self.data.index``.
        - The operation modifies the dataset in place.
        """
        for attr in ("meta_valid", "meta_filled", "filled"):
            frame = getattr(self, attr, None)
            if isinstance(frame, pd.DataFrame):
                if not frame.index.is_unique:
                    frame = frame.loc[~frame.index.duplicated(keep=keep)]
                setattr(self, attr, frame.reindex(self.data.index))
            else:
                setattr(self, attr, pd.DataFrame(index=self.data.index))
    
   

    def _reset_meta_filled_column(self, to_fill: str) -> None:
        """
        Reset filled metadata and values for a column.

        Parameters
        ----------
        to_fill : str
            Name of the column to reset.

        Returns
        -------
        None

        Notes
        -----
        - Resets ``meta_filled[to_fill]`` to match ``meta_valid[to_fill]``.
        - Reinitializes ``filled[to_fill]`` using original data values.
        - Filtered entries are set to NaN.
        - Intended for use when clearing previous filling operations.
        """
        if to_fill not in self.meta_valid.columns:
            self.add_to_meta_valid([to_fill])  
        self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()
        ser = self.data[to_fill].copy()
        ser[self.meta_filled[to_fill] == "filtered"] = np.nan
        self.filled[to_fill] = ser.reindex(self.index())


    def _get_fill_targets(
        self,
        to_fill: str,
        arange: tuple | None = None,
        only_checked: bool = True,
    ) -> pd.Index:
        """
        Get index labels eligible for filling.

        Parameters
        ----------
        to_fill : str
            Name of the column to evaluate.
        arange : tuple, optional
            Range of data to consider. If ``None``, the full dataset is used.
        only_checked : bool, default=True
            If ``True``, restrict selection to rows marked as checked/valid
            for filling.

        Returns
        -------
        pandas.Index
            Index labels corresponding to rows eligible for filling.

        Notes
        -----
        - Only rows currently tagged as ``"filtered"`` in ``meta_filled`` are returned.
        - Previously filled values are not overwritten.
        - The method does not modify the dataset.
        """
        self.meta_filled = self.meta_filled.reindex(self.index())
        if to_fill not in self.meta_filled.columns:
            if to_fill not in self.meta_valid.columns:
                self.add_to_meta_valid([to_fill])
            self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()

        mv = self.meta_filled[to_fill]

        mask = (mv == "filtered")

        if arange is not None:
            try:
                idx_range = self.data.loc[arange[0]:arange[1]].index
            except Exception:
                raise TypeError(
                    "Invalid `arange` bounds for index slicing; use datetime-like or matching index type."
                )
            mask = mask & mv.index.isin(idx_range)


        targets = mv.index[mask]
        return targets
            
#     ###############################################################################
#     ##                          FILLING FUNCTIONS                                ##
#     ###############################################################################
    def fill_missing_interpolation(
        self,
        to_fill: str,
        range_: int,
        arange: Optional[Tuple[object, object]] = None,
        *,
        method: str = "time",           
        limit_direction: str = "both",
        plot: bool = False,
        clear: bool = False,
        **kwargs,
    ) -> None:
        """
        Fill short missing segments using interpolation.

        The method fills filtered values in a column by interpolating over
        short gaps, while preserving original values and longer missing segments.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        range_ : int
            Maximum length of consecutive filtered values to be filled.
            Longer segments remain unfilled.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        method : str, default="time"
            Interpolation method passed to the underlying interpolation routine.
        limit_direction : str, default="both"
            Direction for interpolation (e.g., ``"forward"``, ``"backward"``, ``"both"``).
        plot : bool, default=False
            If ``True``, generate a diagnostic plot of the filling results.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        **kwargs
            Additional keyword arguments passed to the interpolation function.

        Returns
        -------
        None

        Notes
        -----
        - Only filtered values within segments of length ``<= range_`` are filled.
        - Non-filtered values remain unchanged.
        - Filled values are stored in ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` with the tag
        ``"filled_interpol"`` for successfully filled points.
        - Longer filtered segments remain unfilled and retain their original tags.
        - The operation modifies the dataset in place.
        """
        self._plot = "filled"
        self._filling_warning()

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if range_ <= 0:
            raise ValueError("`range_` must be a positive integer.")

        if clear:
            self._reset_meta_filled(to_fill)
        
        self._add_to_meta(to_fill)  

        if arange is not None:
            self._check_rain(arange)

        idx_all = self.data.index

        if arange is None:
            win_idx = idx_all
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(idx_all)
            .fillna("original")
            .replace({"!!": "original"})
        )
        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)

        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        change = filtered_win.ne(filtered_win.shift(1)).cumsum()
        run_lengths = change.map(change.value_counts())
        eligible = filtered_win & (run_lengths <= range_)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        s = pd.to_numeric(self.filled[to_fill], errors="coerce").copy()

        s_win = s.loc[win_idx].copy()
        s_win.loc[eligible] = np.nan

        interp_kwargs = dict(method=method, limit=range_, limit_direction=limit_direction)
        interp_kwargs.update(kwargs)
        try:
            s_win_interp = s_win.interpolate(**interp_kwargs, limit_area="inside")
        except TypeError:
            s_win_interp = s_win.interpolate(**interp_kwargs)

        newly_filled_idx = eligible.index[eligible & s_win_interp.notna()]

        s.loc[win_idx] = s_win_interp
        self.filled[to_fill] = s

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(idx_all)
            .fillna("original")
            .replace({"!!": "original"})
        )
        
        if len(newly_filled_idx) > 0:
            self.meta_filled.loc[newly_filled_idx, to_fill] = "filled_interpol" 

        if plot:
            self.plot_analysed(to_fill)
    

    def fill_missing_kalman(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        model: Literal["local_level", "local_linear_trend"] = "local_level",
        seasonal_periods: Optional[int] = None,  
        max_gap: Optional[int] = None,           
        plot: bool = False,
        clear: bool = False,
        fit_kwargs: Optional[dict] = None,       
    ) -> None:
        """
        Fill missing values using a state-space Kalman smoother.

        The method fills filtered values in a column using estimates from a
        structural time-series model, while preserving original observations.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        model : {"local_level", "local_linear_trend"}, default="local_level"
            Type of structural model used for smoothing.
        seasonal_periods : int, optional
            Length of a deterministic seasonal component, expressed in number of samples.
        max_gap : int, optional
            Maximum length of consecutive filtered values to fill. If ``None``,
            no restriction is applied.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        fit_kwargs : dict, optional
            Additional keyword arguments passed to the model fitting routine.

        Returns
        -------
        None

        Notes
        -----
        - Only values currently tagged as ``"filtered"`` are considered for filling.
        - Original values are preserved.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` with the tag
        ``"filled_kalman"`` where filling is successful.
        - If ``max_gap`` is provided, only filtered runs with length less than or
        equal to ``max_gap`` are filled.
        - The operation modifies the dataset in place.
        """
        import warnings as wn
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)

        self._add_to_meta(to_fill)

        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )

        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)
        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        eligible = filtered_win.copy()
        if max_gap is not None and max_gap > 0:
            change = filtered_win.ne(filtered_win.shift(1)).cumsum()
            run_lengths = change.map(change.value_counts())
            eligible = filtered_win & (run_lengths <= max_gap)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        y_full = pd.to_numeric(self.filled[to_fill], errors="coerce")
        y = y_full.loc[win_idx]

        if isinstance(y.index, pd.DatetimeIndex) and not y.index.is_monotonic_increasing:
            y = y.sort_index()
            eligible = eligible.reindex(y.index, fill_value=False)
            win_idx = y.index  

        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except Exception as e:
            raise ImportError(
                "statsmodels is required for Kalman filling. Install via `pip install statsmodels`."
            ) from e

        level = None
        if model == "local_level":
            level = "llevel"
        elif model == "local_linear_trend":
            level = "ltrend"
        else:
            raise ValueError("`model` must be 'local_level' or 'local_linear_trend'.")

        seasonal = seasonal_periods if (seasonal_periods and seasonal_periods > 1) else None

        ss_mod = UnobservedComponents(
            endog=y,
            level=level,
            seasonal=seasonal,
        )

        fit_kwargs = dict() if fit_kwargs is None else dict(fit_kwargs)
        fit_kwargs.setdefault("disp", False)

        try:
            res = ss_mod.fit(**fit_kwargs)
        except Exception as err:
            wn.warn(f"Kalman fit failed with `{model}`; retrying with local level. Error: {err}")
            ss_mod = UnobservedComponents(endog=y, level="llevel", seasonal=seasonal)
            res = ss_mod.fit(**fit_kwargs)

        try:
            pred = res.get_prediction()
            y_hat = pred.predicted_mean
        except Exception:
            y_hat = res.fittedvalues

        y_hat = pd.to_numeric(y_hat, errors="coerce").reindex(win_idx)

        to_update = eligible.index[eligible & y_hat.notna()]
        if len(to_update) > 0:
            self.filled.loc[to_update, to_fill] = y_hat.loc[to_update]
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            
            self.meta_filled.loc[to_update, to_fill] = "filled_kalman"

        if plot:
            self.plot_analysed(to_fill)

    def fill_missing_arima(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        order: Tuple[int, int, int] = (1, 0, 1),          
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,  
        trend: Optional[Literal["n","c","t","ct"]] = None,
        max_gap: Optional[int] = None,                     
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        plot: bool = False,
        clear: bool = False,
        fit_kwargs: Optional[dict] = None,                 
    ) -> None:
        """
        Fill missing values using ARIMA/SARIMA model predictions.

        The method fills filtered values in a column using in-sample predictions
        from a SARIMAX (ARIMA/SARIMA) model, while preserving original observations.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        order : tuple of int, default=(1, 0, 1)
            Non-seasonal ARIMA order ``(p, d, q)``.
        seasonal_order : tuple of int, optional
            Seasonal ARIMA order ``(P, D, Q, s)``. If ``None``, no seasonal
            component is used.
        trend : {"n", "c", "t", "ct"}, optional
            Trend specification for the SARIMAX model.
        max_gap : int, optional
            Maximum length of consecutive filtered values to fill. If ``None``,
            no restriction is applied.
        enforce_stationarity : bool, default=True
            Whether to enforce stationarity in the ARIMA model.
        enforce_invertibility : bool, default=True
            Whether to enforce invertibility in the ARIMA model.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        fit_kwargs : dict, optional
            Additional keyword arguments passed to the model fitting routine.

        Returns
        -------
        None

        Notes
        -----
        - Only values currently tagged as ``"filtered"`` are considered for filling.
        - Original values are preserved.
        - Predictions are generated using a SARIMAX model fitted on the selected data.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` with the tag
        ``"filled_arima"`` where filling is successful.
        - If ``max_gap`` is provided, only filtered runs with length less than or
        equal to ``max_gap`` are filled.
        - The operation modifies the dataset in place.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)

        self._add_to_meta(to_fill)

        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        # tags_valid = (
        #     self.meta_valid[to_fill]
        #     .reindex(self.data.index)
        #     .fillna("original")
        #     .replace({"!!": "original"})
        # )
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )

        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)
        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        eligible = filtered_win.copy()
        if max_gap is not None and max_gap > 0:
            change = filtered_win.ne(filtered_win.shift(1)).cumsum()
            run_lengths = change.map(change.value_counts())
            eligible = filtered_win & (run_lengths <= max_gap)
        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        y_full = pd.to_numeric(self.filled[to_fill], errors="coerce")
        y = y_full.loc[win_idx]

        if isinstance(y.index, pd.DatetimeIndex) and not y.index.is_monotonic_increasing:
            y = y.sort_index()
            eligible = eligible.reindex(y.index, fill_value=False)
            win_idx = y.index

        if isinstance(win_idx, pd.DatetimeIndex) and seasonal_order is not None and win_idx.freq is None:
            try:
                win_idx = win_idx.inferred_freq and win_idx
            except Exception:
                pass  

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except Exception as e:
            raise ImportError("statsmodels is required for ARIMA filling. `pip install statsmodels`.") from e

        fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        fit_kwargs.setdefault("disp", False)

        model = SARIMAX(
            endog=y,
            order=order,
            seasonal_order=(seasonal_order or (0, 0, 0, 0)),
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        try:
            res = model.fit(**fit_kwargs)
        except Exception as err:
            try:
                model2 = SARIMAX(endog=y, order=(max(order[0],1), 0, 0), seasonal_order=(0,0,0,0), trend=None)
                res = model2.fit(disp=False)
            except Exception as err2:
                raise RuntimeError(f"SARIMAX fit failed: {err} / fallback: {err2}")

        try:
            pred = res.get_prediction()
            y_hat = pred.predicted_mean
        except Exception:
            y_hat = res.fittedvalues
        y_hat = pd.to_numeric(y_hat, errors="coerce").reindex(win_idx)

        to_update = eligible.index[eligible & y_hat.notna()]
        if len(to_update) > 0:
            self.filled.loc[to_update, to_fill] = y_hat.loc[to_update]
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            self.meta_filled.loc[to_update, to_fill] = "filled_arima"

        if plot:
            self.plot_analysed(to_fill)

    def fill_missing_gaussian(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,   
        *,
        only_checked: bool = True,
        clear: bool = False,
        plot: bool = False,
        max_train_points: int = 5000,
        stride: int = 1,
        kernel: Optional[object] = None,
        seasonal_period: Optional[Union[pd.Timedelta, float]] = None,  
        unit: str = "d",                          
        alpha: Optional[float] = None,            
        normalize_y: bool = True,
        n_restarts_optimizer: int = 2,
        random_state: Optional[int] = None,
        context_expand: Optional[pd.Timedelta] = None,  
        X_cols: Optional[Sequence[str]] = None,  
    ) -> None:
        """
        Fill missing values using Gaussian Process regression.

        The method fits a Gaussian Process (GP) model on observed data and predicts
        values at filtered indices, optionally incorporating exogenous variables.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        arange : tuple, optional
            Range of data over which the model is trained and applied. If ``None``,
            the full dataset is used.
        only_checked : bool, default=True
            If ``True``, restrict predictions to indices marked as checked/valid
            for filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        max_train_points : int, default=5000
            Maximum number of training points used for GP fitting.
        stride : int, default=1
            Subsampling step applied to training data to reduce computational cost.
        kernel : object, optional
            Custom kernel for the Gaussian Process model.
        seasonal_period : pandas.Timedelta or float, optional
            Period of a seasonal component to include via a periodic kernel.
        unit : str, default="d"
            Unit used when interpreting numeric time values.
        alpha : float, optional
            Value added to the diagonal of the kernel matrix for numerical stability.
        normalize_y : bool, default=True
            Whether to normalize the target variable before fitting.
        n_restarts_optimizer : int, default=2
            Number of restarts for kernel hyperparameter optimization.
        random_state : int, optional
            Random seed for reproducibility.
        context_expand : pandas.Timedelta, optional
            Optional extension of the training window around ``arange``.
        X_cols : sequence of str, optional
            Additional columns used as exogenous input features.

        Returns
        -------
        None

        Notes
        -----
        - The model is trained on points tagged as ``"original"`` in ``meta_valid``.
        - Predictions are made for indices tagged as ``"filtered"``.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` with the tag
        ``"filled_gaussian"`` where filling is successful.
        - Gaussian Process training scales cubically with the number of points;
        use ``max_train_points`` and ``stride`` to control computational cost.
        - A periodic kernel can be included via ``seasonal_period`` to capture
        repeating patterns.
        - The operation modifies the dataset in place.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)

        if arange is None:
            win_idx = self.data.index
            train_slice = slice(self.data.index.min(), self.data.index.max())
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` must slice the DataFrame index.") from e
            train_slice = slice(arange[0], arange[1])

        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if only_checked:
            mv = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            # mv = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            target_idx = win_idx[mv.loc[win_idx].eq("filtered")]
        else:
            target_idx = win_idx

        if len(target_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if isinstance(self.data.index, pd.DatetimeIndex) and context_expand is not None and arange is not None:
            start = pd.to_datetime(arange[0]) - context_expand
            end   = pd.to_datetime(arange[1]) + context_expand
            train_slice = slice(start, end)

        y_full = pd.to_numeric(self.data[to_fill], errors="coerce")
        y_train = y_full.loc[train_slice].copy()
        if hasattr(self, "meta_valid") and to_fill in self.meta_valid:
            mv_train = (
                self.meta_valid[to_fill]
                .reindex(y_train.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            y_train = y_train.where(mv_train.eq("original"))

        y_train = y_train.dropna()
        if stride > 1 and len(y_train) > 0:
            y_train = y_train.iloc[::stride]

        if len(y_train) > max_train_points:
            take = np.linspace(0, len(y_train) - 1, max_train_points).astype(int)
            y_train = y_train.iloc[take]
            wn.warn(
                f"GP training truncated to {max_train_points} points (from {len(y_full.loc[train_slice])}).",
                RuntimeWarning, stacklevel=2
            )

        if len(y_train) < 5:
            wn.warn("Not enough original samples for GP training.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        anchor_idx = y_train.index.union(target_idx)

        def _to_numeric_time(
            ix: pd.Index,
            origin: Union[pd.Timestamp, float],
            is_datetime: bool,
            unit: str = "d",
        ) -> np.ndarray:
            """
            Convert an index to a strictly NumPy float array of elapsed time in the requested unit.
            Works reliably across pandas versions.
            """
            if is_datetime:
                di = pd.DatetimeIndex(ix)
                secs = (di - pd.Timestamp(origin)) / np.timedelta64(1, "s")
                arr = np.asarray(secs, dtype=float)
                if unit in ("sec", "s"):
                    return arr
                if unit in ("min", "m"):
                    return arr / 60.0
                if unit in ("hr", "h"):
                    return arr / 3600.0
                return arr / 86400.0
            else:
                return np.asarray(pd.Index(ix).to_numpy(dtype=float), dtype=float)

        is_dt = isinstance(anchor_idx, pd.DatetimeIndex)
        if is_dt:
            origin = anchor_idx.min()
            # x_train_time = _to_numeric_time(y_train.index, origin, True).reshape(-1, 1)
            # x_target_time = _to_numeric_time(target_idx, origin, True).reshape(-1, 1)
        else:
            origin = 0.0
            # x_train_time = _to_numeric_time(y_train.index, origin, False).reshape(-1, 1)
            # x_target_time = _to_numeric_time(target_idx, origin, False).reshape(-1, 1)
        
        x_train_time = _to_numeric_time(y_train.index, origin, is_dt, unit).reshape(-1, 1)
        x_target_time = _to_numeric_time(target_idx, origin, is_dt, unit).reshape(-1, 1)

        X_train_list = [x_train_time]
        X_target_list = [x_target_time]

        if X_cols:
            for col in X_cols:
                if col not in self.data.columns:
                    raise KeyError(f"Exogenous column '{col}' not found.")
                ex = pd.to_numeric(self.data[col], errors="coerce")
                X_train_list.append(ex.reindex(y_train.index).to_numpy().reshape(-1, 1))
                X_target_list.append(ex.reindex(target_idx).to_numpy().reshape(-1, 1))

        X_train = np.concatenate(X_train_list, axis=1)
        X_target = np.concatenate(X_target_list, axis=1)

        mask_ok = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train.to_numpy())
        X_train = X_train[mask_ok]
        y_train = y_train.iloc[mask_ok]

        if len(y_train) < 5:
            wn.warn("Not enough valid rows after assembling GP training matrix.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared
        except ImportError as e:
            raise ImportError("scikit-learn is required for fill_missing_gaussian. Install with `pip install scikit-learn`.") from e

        if kernel is None:
            k_base = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            if seasonal_period is not None:
                if isinstance(seasonal_period, pd.Timedelta):
                    period_sec = seasonal_period.total_seconds()
                else:
                    period_sec = float(seasonal_period)  
                if is_dt:
                    if unit in ("sec","s"):      p = period_sec
                    elif unit in ("min","m"):    p = period_sec / 60.0
                    elif unit in ("hr","h"):     p = period_sec / 3600.0
                    else:                        p = period_sec / 86400.0
                else:
                    p = period_sec  
                k_season = ExpSineSquared(length_scale=1.0, periodicity=p,
                                        periodicity_bounds=(max(1e-6, p/10), p*10))
                k = ConstantKernel(1.0, (1e-3, 1e3)) * (k_base + k_season) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            else:
                k = ConstantKernel(1.0, (1e-3, 1e3)) * k_base + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        else:
            k = kernel

        gp = GaussianProcessRegressor(
            kernel=k,
            alpha=0.0 if alpha is None else float(alpha),
            normalize_y=normalize_y,
            n_restarts_optimizer=int(n_restarts_optimizer),
            random_state=random_state,
        )

        try:
            gp.fit(X_train, y_train.to_numpy().astype(float))
        except Exception as e:
            wn.warn(f"GP fitting failed: {e}", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        try:
            y_pred, y_std = gp.predict(X_target, return_std=True)
        except Exception as e:
            wn.warn(f"GP prediction failed: {e}", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        y_pred = pd.Series(pd.to_numeric(y_pred, errors="coerce"), index=target_idx).dropna()
        if y_pred.empty:
            if plot:
                self.plot_analysed(to_fill)
            return

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[y_pred.index, to_fill] = y_pred.values
        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[y_pred.index, to_fill] = "filled_gaussian"

        if plot:
            self.plot_analysed(to_fill)


    

    def fill_missing_ratio(
        self,
        to_fill: str,
        to_use: str,
        ratio: Optional[float] = None,
        arange: Optional[Tuple[object, object]] = None,
        *,
        intercept: float = 0.0,
        estimate_params: bool = False,
        train_range: Optional[Tuple[object, object]] = None,
        train_only_original: bool = True,
        zero_intercept: bool = False,
        robust: bool = False,
        min_train_points: int = 20,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
        return_params: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Fill missing values using a linear relationship with another column.

        The method fills filtered values in a column using a linear relation of the form
        ``new = ratio * data[to_use] + intercept``, with optional parameter estimation.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        to_use : str
            Name of the reference (driver) column.
        ratio : float, optional
            Multiplicative factor. Required if ``estimate_params=False``.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        intercept : float, default=0.0
            Additive constant applied in the linear relation.
        estimate_params : bool, default=False
            If ``True``, estimate ``ratio`` and ``intercept`` from data.
        train_range : tuple, optional
            Range used to estimate parameters. Defaults to ``arange`` if provided,
            otherwise the full dataset.
        train_only_original : bool, default=True
            If ``True``, use only points tagged as ``"original"`` for training.
        zero_intercept : bool, default=False
            If ``True``, force the intercept to zero during estimation.
        robust : bool, default=False
            If ``True``, use a robust regression method (e.g., RANSAC) if available.
        min_train_points : int, default=20
            Minimum number of samples required to estimate parameters.
        only_checked : bool, default=True
            If ``True``, restrict filling to indices tagged as ``"filtered"``.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        return_params : bool, default=False
            If ``True``, return the fitted or used parameters.

        Returns
        -------
        dict or None
            If ``return_params=True``, returns a dictionary with keys
            ``{"ratio", "intercept"}``. Otherwise, returns ``None``.

        Notes
        -----
        - If ``estimate_params=True``, parameters are fitted using the specified
        training range and constraints.
        - Only values currently tagged as ``"filtered"`` are filled (unless
        ``only_checked=False``).
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` to reflect filled values.
        - The dataset is modified in place.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        for col in (to_fill, to_use):
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  

        def _slice_index(bounds: Optional[Tuple[object, object]]) -> pd.Index:
            if bounds is None:
                return self.data.index
            try:
                return self.data.loc[bounds[0]:bounds[1]].index
            except Exception as e:
                raise TypeError("Bounds must match index dtype.") from e

        win_idx = _slice_index(arange)
        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return None

        train_idx = _slice_index(train_range if train_range is not None else arange)
        if len(train_idx) == 0:
            train_idx = self.data.index  

        used_ratio = ratio
        used_intercept = 0.0 if zero_intercept and estimate_params else intercept

        if estimate_params:
            # Choose training mask: points with valid pairs (to_fill & to_use)
            y_train = pd.to_numeric(self.data[to_fill].reindex(train_idx), errors="coerce")
            x_train = pd.to_numeric(self.data[to_use].reindex(train_idx), errors="coerce")

            if train_only_original:
                tags = (
                    self.meta_valid[to_fill]
                    .reindex(self.data.index)
                    .fillna("original")
                    .replace({"!!": "original"})
                )
                orig_mask = tags.reindex(train_idx, fill_value="original").eq("original")
            else:
                orig_mask = pd.Series(True, index=train_idx)

            mask = orig_mask & y_train.notna() & x_train.notna()
            x = x_train[mask].astype(float).values.reshape(-1, 1)
            y = y_train[mask].astype(float).values

            if x.shape[0] < max(2, min_train_points):
                wn.warn(
                    f"Not enough training points for estimation (have {x.shape[0]}, need ≥ {min_train_points}). "
                    "Falling back to provided ratio/intercept.",
                    RuntimeWarning,
                )
            else:
                try:
                    if robust:
                        from sklearn.linear_model import RANSACRegressor, LinearRegression  
                        base = LinearRegression(fit_intercept=not zero_intercept)
                        ransac = RANSACRegressor(base_estimator=base, random_state=0)
                        ransac.fit(x, y)
                        coef = getattr(ransac.estimator_, "coef_", np.array([np.nan]))[0]
                        intercept_est = getattr(ransac.estimator_, "intercept_", 0.0)
                        used_ratio = float(coef)
                        used_intercept = 0.0 if zero_intercept else float(intercept_est)
                    else:
                        if zero_intercept:
                            # y = a * x (no intercept)
                            denom = float(np.dot(x.ravel(), x.ravel()))
                            used_ratio = float(np.dot(x.ravel(), y) / denom) if denom != 0.0 else 0.0
                            used_intercept = 0.0
                        else:
                            # y = a * x + b
                            a, b = np.polyfit(x.ravel(), y, 1)
                            used_ratio = float(a)
                            used_intercept = float(b)
                except Exception as e:
                    wn.warn(f"Parameter estimation failed ({e}). Falling back to provided ratio/intercept.", RuntimeWarning)

        if used_ratio is None or not np.isfinite(used_ratio):
            raise ValueError("`ratio` must be provided or successfully estimated.")
        if not np.isfinite(used_intercept):
            used_intercept = 0.0

        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return {"ratio": used_ratio, "intercept": used_intercept} if return_params else None

        x_driver = pd.to_numeric(self.data[to_use].reindex(win_idx), errors="coerce")
        replacements = used_ratio * x_driver + used_intercept

        target_idx = eligible.index[eligible & replacements.notna()]
        if len(target_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return {"ratio": used_ratio, "intercept": used_intercept} if return_params else None

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[target_idx, to_fill] = replacements.loc[target_idx]

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[target_idx, to_fill] = "filled_ratio"

        if plot:
            self.plot_analysed(to_fill)

        if return_params:
            return {"ratio": used_ratio, "intercept": used_intercept}
        return None



    def fill_missing_standard(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
    ) -> None:
        """
        Fill missing values using a standard daily profile.

        The method fills filtered values in a column using the average daily
        profile computed from the data.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        only_checked : bool, default=True
            If ``True``, restrict filling to indices tagged as ``"filtered"``.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.

        Returns
        -------
        None

        Notes
        -----
        - Filling is based on the average daily profile computed by
        ``calc_daily_profile()``.
        - Values are matched by time-of-day.
        - Original values are preserved.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` with the tag
        ``"filled_average_profile"`` where filling is successful.
        - The dataset is modified in place.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if to_fill not in getattr(self, "daily_profile", {}):
            raise KeyError(
                f"No daily profile found for '{to_fill}'. Run calc_daily_profile() first."
            )

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  

        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        prof = self.daily_profile[to_fill][["avg"]].copy()
        td = pd.to_timedelta(pd.Index(prof.index.astype(str)), errors="coerce")
        prof = prof.assign(secs=pd.Series(td.total_seconds(), index=prof.index))
        prof = prof.dropna(subset=["secs", "avg"]).sort_values("secs")
        if prof.empty:
            wn.warn("Daily profile has no valid time-of-day entries.", RuntimeWarning, stacklevel=2)
            return

        prof_secs = prof["secs"].to_numpy(dtype=float)
        prof_vals = pd.to_numeric(prof["avg"], errors="coerce").to_numpy(dtype=float)
        mask_ok = np.isfinite(prof_secs) & np.isfinite(prof_vals)
        prof_secs = prof_secs[mask_ok]
        prof_vals = prof_vals[mask_ok]
        if prof_secs.size == 0:
            wn.warn("Daily profile contains no finite (secs, avg) pairs.", RuntimeWarning, stacklevel=2)
            return

        elig_idx = eligible.index[eligible]
        if isinstance(self.data.index, pd.DatetimeIndex):
            td_since_midnight = elig_idx - elig_idx.normalize()
            tgt_secs = td_since_midnight.total_seconds().astype(float)
        else:
            vals = pd.Index(elig_idx).astype(float)
            frac = vals - np.floor(vals)
            tgt_secs = (frac * 86400.0).astype(float)

        filled_vals = np.interp(
            tgt_secs,
            prof_secs,
            prof_vals,
            left=np.nan,
            right=np.nan,
        )
        ok = np.isfinite(filled_vals)
        if not ok.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        to_write_idx = elig_idx[ok]
        to_write_vals = filled_vals[ok]

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[to_write_idx, to_fill] = to_write_vals

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[to_write_idx, to_fill] = "filled_average_profile"

        if plot:
            self.plot_analysed(to_fill)
    

    

    def fill_missing_model(
        self,
        to_fill: str,
        to_use: Union[pd.Series, pd.DataFrame],
        arange: Optional[Tuple[object, object]] = None,
        *,
        only_checked: bool = True,
        unit: str = "d",
        plot: bool = False,
        clear: bool = False,
        tolerance: Optional[Union[pd.Timedelta, float]] = None,
    ) -> None:
        """
        Fill missing values using an external model time series.

        The method fills filtered values in a column using values from an
        external series or model output, aligned to the dataset index.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        to_use : pandas.Series or pandas.DataFrame
            External time series providing values for filling.
        arange : tuple, optional
            Range of data over which filling is applied. If ``None``, the full
            dataset is considered.
        only_checked : bool, default=True
            If ``True``, restrict filling to indices tagged as ``"filtered"``.
        unit : str, default="d"
            Unit used when interpreting numeric time values for alignment.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.
        tolerance : pandas.Timedelta or float, optional
            Maximum allowed difference when aligning external data to the dataset index.

        Returns
        -------
        None

        Notes
        -----
        - External data is aligned to the dataset index before filling.
        - Only values currently tagged as ``"filtered"`` are filled (unless
        ``only_checked=False``).
        - Original values are preserved.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` to reflect filled values.
        - The dataset is modified in place.
        """
        
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if isinstance(to_use, pd.DataFrame):
            if to_use.shape[1] != 1:
                raise ValueError("`to_use` DataFrame must have exactly one column.")
            model_series = to_use.iloc[:, 0].copy()
        elif isinstance(to_use, pd.Series):
            model_series = to_use.copy()
        else:
            raise TypeError("`to_use` must be a pandas Series or single-column DataFrame.")
        model_series = pd.to_numeric(model_series, errors="coerce")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)

        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)
        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        def _to_relative_numeric(tidx: pd.Index, origin: Optional[pd.Timestamp] = None) -> np.ndarray:
            if not isinstance(tidx, pd.DatetimeIndex):
                raise TypeError("_to_relative_numeric expects a DatetimeIndex.")
            if origin is None:
                origin = tidx.min().normalize()
            delta = (tidx - origin)
            secs = delta.total_seconds()
            if unit in ("sec", "s"):
                return secs
            if unit in ("min", "m"):
                return secs / 60.0
            if unit in ("hr", "h"):
                return secs / 3600.0
            if unit in ("d", "day", "days"):
                return secs / 86400.0
            raise ValueError("`unit` must be one of {'sec','min','hr','d'}.")

        data_idx = self.data.index
        left_idx = eligible.index[eligible]
        left_is_dt = isinstance(data_idx, pd.DatetimeIndex)
        right_is_dt = isinstance(model_series.index, pd.DatetimeIndex)

        def _with_orig_index(df: pd.DataFrame, name="_orig_index") -> pd.DataFrame:
            out = df.copy()
            out[name] = out.index
            return out.reset_index(drop=True)

        if left_is_dt and right_is_dt:
            left_df = pd.DataFrame({"ts": pd.to_datetime(left_idx)}, index=left_idx)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("ts")

            right_df = pd.DataFrame(
                {"ts": pd.to_datetime(model_series.index), "yhat": model_series.values}
            ).sort_values("ts")

            tol = tolerance if (isinstance(tolerance, pd.Timedelta) or tolerance is None) else pd.to_timedelta(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="ts", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        elif (not left_is_dt) and (not right_is_dt):
            left_vals = pd.Series(left_idx, index=left_idx).astype(float)
            left_df = pd.DataFrame({"x": left_vals.values}, index=left_vals.index)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("x")

            right_vals = pd.Series(model_series.index).astype(float)
            right_df = pd.DataFrame({"x": right_vals.values, "yhat": model_series.values}).sort_values("x")

            tol = None if tolerance is None else float(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="x", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        else:
            if left_is_dt:
                origin = pd.to_datetime(
                    min(left_idx.min(),
                        model_series.index.min() if right_is_dt else pd.Timestamp.utcnow())
                )
                left_x = _to_relative_numeric(pd.DatetimeIndex(left_idx), origin=origin)
                if right_is_dt:
                    right_x = _to_relative_numeric(pd.DatetimeIndex(model_series.index), origin=origin)
                else:
                    right_x = pd.Index(model_series.index).astype(float).to_numpy()
            else:
                origin = pd.to_datetime(
                    min(model_series.index.min(),
                        left_idx.min() if left_is_dt else pd.Timestamp.utcnow())
                )
                right_x = _to_relative_numeric(pd.DatetimeIndex(model_series.index), origin=origin)
                left_x = pd.Index(left_idx).astype(float).to_numpy()

            left_df = pd.DataFrame({"x": left_x}, index=left_idx)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("x")
            right_df = pd.DataFrame({"x": right_x, "yhat": model_series.values}).sort_values("x")

            tol = None
            if tolerance is not None:
                if isinstance(tolerance, pd.Timedelta):
                    total_sec = tolerance.total_seconds()
                    tol = {"sec": total_sec, "s": total_sec,
                        "min": total_sec/60, "m": total_sec/60,
                        "hr": total_sec/3600, "h": total_sec/3600,
                        "d": total_sec/86400, "day": total_sec/86400, "days": total_sec/86400}[unit]
                else:
                    tol = float(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="x", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        yhat = pd.to_numeric(yhat, errors="coerce")
        valid_targets = yhat.index[yhat.notna()]
        if len(valid_targets) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[valid_targets, to_fill] = yhat.loc[valid_targets].values

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[valid_targets, to_fill] = "filled_model"

        if plot:
            self.plot_analysed(to_fill)

   

    def fill_missing_daybefore(
        self,
        to_fill: str,
        arange: Tuple[object, object],
        range_to_replace: List[float] = [1.0, 4.0],  
        *,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
    ) -> None:
        """
        Fill missing values using data from the previous day.

        The method fills filtered values in a column by copying values from the
        same time-of-day on the previous day, using the best available source.

        Parameters
        ----------
        to_fill : str
            Name of the column to fill.
        arange : tuple
            Range of data over which filling is applied. The range must begin at
            least one day after the start of the dataset.
        range_to_replace : list of float, default=[1.0, 4.0]
            Minimum and maximum consecutive gap length, expressed in days, that
            will be filled. These values are converted to numbers of samples
            based on the previous-day sample count.
        only_checked : bool, default=True
            If ``True``, restrict filling to indices tagged as ``"filtered"``.
            Otherwise, consider all rows in ``arange``.
        plot : bool, default=False
            If ``True``, generate a diagnostic plot after filling.
        clear : bool, default=False
            If ``True``, reset previous filling results before applying the method.

        Returns
        -------
        None

        Notes
        -----
        - The method requires equidistant sampling.
        - Values are copied from the same time-of-day on the previous day.
        - When available, values from ``self.filled[to_fill]`` are used first;
        otherwise, values from ``self.data[to_fill]`` are used.
        - Original values are preserved.
        - Filled values are written to ``self.filled[to_fill]``.
        - Metadata is updated in ``self.meta_filled[to_fill]`` to reflect filled values.
        - The dataset is modified in place.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if arange is None or len(arange) != 2:
            raise ValueError("`arange` must be a (start, end) tuple; this method requires a bounded window.")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  

        idx = self.data.index
        start, end = arange
        try:
            win_idx = self.data.loc[start:end].index
        except Exception as e:
            raise TypeError("`arange` bounds must be sliceable on the current index.") from e
        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if isinstance(idx, pd.DatetimeIndex):
            start_ts = pd.to_datetime(start)
            if start_ts - pd.Timedelta(days=1) < idx.min():
                raise IndexError("No previous-day data available; choose a later `arange` start.")
            prev_slice = slice(start_ts - pd.Timedelta(days=1), start_ts)
        else:
            start_num = float(start)
            if start_num - 1.0 < float(idx.min()):
                raise IndexError("No previous-day data available; choose a later `arange` start.")
            prev_slice = slice(start_num - 1.0, start_num)

        src_filled = self.filled.get(to_fill, pd.Series(index=self.data.index, dtype=float)).loc[prev_slice]
        use_filled = src_filled.notna().any()

        if use_filled:
            prev_series = src_filled
        else:
            src_data = pd.to_numeric(self.data[to_fill], errors="coerce").loc[prev_slice].copy()
            if hasattr(self, "meta_valid") and to_fill in self.meta_valid:
                mv_prev = self.meta_valid[to_fill].reindex(src_data.index)
                prev_series = src_data.where(mv_prev.eq("original"))
                if prev_series.dropna().empty:
                    prev_series = src_data
            else:
                prev_series = src_data

        if prev_series.dropna().empty:
            raise ValueError("Previous-day window has no usable samples to form a profile.")

        if isinstance(idx, pd.DatetimeIndex):
            prev_series = prev_series.dropna()
            prev_key = prev_series.index.time
            day_before = pd.DataFrame({"data": prev_series.values}, index=pd.Index(prev_key, name="tod"))
            day_before = day_before[~day_before.index.duplicated(keep="first")]
            day_size = len(day_before)
        else:
            prev_series = prev_series.dropna()
            prev_vals = pd.Index(prev_series.index).astype(float).to_numpy()
            frac_prev = (prev_vals - np.floor(prev_vals))
            order = np.argsort(frac_prev)
            frac_sorted = frac_prev[order]
            vals_sorted = prev_series.values[order]
            _, uniq_idx = np.unique(frac_sorted, return_index=True)
            day_before = pd.DataFrame({"frac": frac_sorted[uniq_idx], "data": vals_sorted[uniq_idx]}) \
                            .set_index("frac").sort_index()
            day_size = len(day_before)

        if day_size == 0:
            raise ValueError("Previous-day profile has no valid samples.")

        min_pts = int(np.floor(range_to_replace[0] * day_size))
        max_pts = int(np.floor(range_to_replace[1] * day_size))
        if max_pts < 1:
            wn.warn("`range_to_replace` converts to <1 point; nothing will be filled.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        if only_checked:
            # mv = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            mv = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            cand = mv.loc[start:end].eq("filtered")
        else:
            cand = pd.Series(True, index=win_idx)

        if not cand.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        labels = (
            self.meta_valid[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
            .loc[start:end]
        )
        grp_ids = (labels != labels.shift()).cumsum()
        run_sizes = grp_ids.map(grp_ids.value_counts())
        mask_len_ok = (run_sizes >= min_pts) & (run_sizes <= max_pts)
        to_replace_idx = labels.index[cand & mask_len_ok]
        if len(to_replace_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if isinstance(idx, pd.DatetimeIndex):
            target_tod = pd.Index(to_replace_idx).time
            src = pd.Series(day_before["data"].values, index=day_before.index, dtype=float)
            fill_vals = pd.Series(index=to_replace_idx, dtype=float)
            fill_vals.loc[:] = pd.Series(target_tod, index=to_replace_idx).map(src).to_numpy()
        else:
            tgt_vals = pd.Index(to_replace_idx).astype(float).to_numpy()
            tgt_frac = (tgt_vals - np.floor(tgt_vals))
            src_x = day_before.index.to_numpy(dtype=float)
            src_y = day_before["data"].to_numpy(dtype=float)
            fill_arr = np.interp(tgt_frac, src_x, src_y, left=np.nan, right=np.nan)
            fill_vals = pd.Series(fill_arr, index=to_replace_idx, dtype=float)

        ok = np.isfinite(fill_vals.values)
        if not ok.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        valid_idx = fill_vals.index[ok]
        valid_vals = fill_vals.values[ok]

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[valid_idx, to_fill] = valid_vals
        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[valid_idx, to_fill] = "filled_profile_day_before"

        if plot:
            self.plot_analysed(to_fill)




#     ###############################################################################
#     ##                          RELIABILITY FUNCTIONS                            ##
#     ###############################################################################
    def _create_gaps(
        self,
        data_name: str,
        range_: tuple,
        number: int,
        max_size: int,
        *,
        reset: bool = False,
        user_output: bool = False,
        random_state: Optional[int] = None,
    ) -> pd.Index:
        """
        Create artificial gaps in a data column.

        Parameters
        ----------
        data_name : str
            Name of the column where gaps are introduced.
        range_ : tuple
            Range of data within which gaps are created.
        number : int
            Number of gap segments to generate.
        max_size : int
            Maximum length of each gap (in number of samples).
        reset : bool, default=False
            If ``True``, reset existing gap markings before creating new ones.
        user_output : bool, default=False
            If ``True``, provide additional output or feedback to the user.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pandas.Index
            Index labels corresponding to the created gaps.

        Notes
        -----
        - Gaps are created by tagging rows as ``"filtered"`` in ``meta_valid[data_name]``.
        - Values in the specified column are set to 0 at the gap indices.
        - The method modifies the dataset in place.
        """
       

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        if reset:
            self._reset_meta_valid(data_name)

        if not hasattr(self, "meta_valid") or not isinstance(self.meta_valid, pd.DataFrame):
            self.meta_valid = pd.DataFrame(index=self.data.index)
        self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.meta_valid[data_name] = "original"

        try:
            window = self.data.loc[range_[0]:range_[1]]
        except Exception as e:
            raise TypeError(
                "Slicing not possible for given index type and range_. "
                "Ensure range_ matches the index label type."
            ) from e

        if window.empty:
            raise ValueError("Selected `range_` yields an empty window; adjust the bounds.")

        idx_all = self.data.index
        pos_window = idx_all.get_indexer(window.index)
        pos_window = pos_window[pos_window >= 0]

        if number <= 0 or max_size <= 0 or len(pos_window) < 2:
            return pd.Index([])

        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        low_val = int(pos_window.min())
        high_val = int(pos_window.max())

        starts = rng.randint(low_val, high_val, size=number)

        lengths = rng.randint(1, max_size + 1, size=number)

        locs_list = [np.arange(s, s + L, dtype=int) for s, L in zip(starts, lengths)]
        if not locs_list:
            return pd.Index([])

        locs = np.unique(np.clip(np.concatenate(locs_list), low_val, high_val))
        gap_index = idx_all[locs]

        self.data.loc[gap_index, data_name] = 0
        self.meta_valid.loc[gap_index, data_name] = "filtered"

        if user_output:
            counts = self.meta_valid[data_name].value_counts(dropna=False)
            n_total = len(self.meta_valid)
            n_orig = int(counts.get("original", 0))
            left_pct = (n_orig * 100.0) / max(1, n_total)
            print(f"{left_pct:.2f}% of datapoints left after creating gaps")

        return pd.Index(gap_index)
   

    def _calculate_filling_error(
        self,
        data_name: str,
        filling_function: Union[str, Callable[..., Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        **options: Dict[str, Any],
    ) -> Optional[float]:
        """
        Evaluate filling performance using artificial gaps.

        Parameters
        ----------
        data_name : str
            Name of the column to evaluate.
        filling_function : str or callable
            Filling method to apply. Can be the name of a method or a callable.
        test_data_range : tuple
            Range of data over which the evaluation is performed.
        nr_small_gaps : int, default=0
            Number of small gaps to generate.
        max_size_small_gaps : int, default=0
            Maximum size of small gaps (in number of samples).
        nr_large_gaps : int, default=0
            Number of large gaps to generate.
        max_size_large_gaps : int, default=0
            Maximum size of large gaps (in number of samples).
        random_state : int, optional
            Random seed for reproducibility.
        **options : dict
            Additional keyword arguments passed to the filling function.

        Returns
        -------
        float or None
            Percentage error between filled and original values. Returns ``None``
            if the evaluation cannot be performed.

        Notes
        -----
        - Artificial gaps are created in a copy of the dataset.
        - The specified filling method is applied to reconstruct missing values.
        - The filled values are compared to the original data to compute error.
        - The dataset itself is not modified.
        """
        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found.")
        
        if "to_fill" not in options or options["to_fill"] is None:
            options["to_fill"] = data_name

        if "arange" not in options or not isinstance(options["arange"], (tuple, list)) or len(options["arange"]) != 2:
            # raise ValueError("`options` must include 'arange' = (start, end).")
            options['arange'] = (self.data.index[0], self.data.index[-1])

        to_fill: str = options.get("to_fill", data_name)

        s, e = test_data_range
        try:
            orig = self.__class__(self.data.loc[s:e].copy(),
                                timedata_column=getattr(self, "timename", "index"),
                                data_type=getattr(self, "data_type", None),
                                experiment_tag=getattr(self, "tag", None),
                                time_unit=getattr(self, "time_unit", None))
            gaps = self.__class__(self.data.loc[s:e].copy(),
                                timedata_column=getattr(self, "timename", "index"),
                                data_type=getattr(self, "data_type", None),
                                experiment_tag=getattr(self, "tag", None),
                                time_unit=getattr(self, "time_unit", None))
        except Exception as ex:
            raise TypeError("`test_data_range` does not align with index labels.") from ex

        if orig.data.empty:
            return None

        for attr in ("meta_valid", "meta_filled"):
            frame = getattr(gaps, attr, None)
            if not isinstance(frame, pd.DataFrame):
                frame = pd.DataFrame(index=gaps.data.index)
            else:
                frame = frame.reindex(gaps.data.index)
            setattr(gaps, attr, frame)

        if to_fill not in gaps.meta_valid.columns:
            gaps.meta_valid[to_fill] = "original"
        if to_fill not in gaps.meta_filled.columns:
            gaps.meta_filled[to_fill] = gaps.meta_valid[to_fill].copy()

        try:
            gaps.get_highs(data_name, 0.9, [s, e])
        except Exception:
            pass

        tagged_all = pd.Index([])
        a_start, a_end = options["arange"]
        if nr_small_gaps > 0:
            idx_small = gaps._create_gaps(
                data_name, (a_start, a_end), nr_small_gaps, max_size_small_gaps,
                reset=True, user_output=False, random_state=random_state
            )
            tagged_all = tagged_all.union(idx_small)
        if nr_large_gaps > 0:
            idx_large = gaps._create_gaps(
                data_name, (a_start, a_end), nr_large_gaps, max_size_large_gaps,
                reset=False, user_output=False, random_state=None if random_state is None else random_state + 1
            )
            tagged_all = tagged_all.union(idx_large)

        if tagged_all.empty:
            return None

        gaps.filled = gaps.data.apply(pd.to_numeric, errors="coerce").copy()
        gaps.filled.loc[tagged_all, to_fill] = np.nan
        gaps.meta_filled[to_fill] = gaps.meta_valid[to_fill].copy()

        if callable(filling_function):
            filling_function(gaps, **options)
        else:
            if not hasattr(gaps, filling_function):
                raise ValueError(f"Filling method '{filling_function}' not found.")
            getattr(gaps, filling_function)(**options)

        if to_fill in gaps.meta_filled.columns:
            mf = gaps.meta_filled[to_fill].astype(str)
            filled_idx = mf.index[mf.str.startswith("filled_")]
            score_idx = tagged_all.intersection(filled_idx)
        else:
            score_idx = pd.Index([])

        if score_idx.empty:
            non_nan = gaps.filled.loc[tagged_all, to_fill].dropna().index
            score_idx = non_nan

        if score_idx.empty:
            return None

        o = pd.to_numeric(orig.data.loc[score_idx, to_fill], errors="coerce").astype(float)
        p = pd.to_numeric(gaps.filled.loc[score_idx, to_fill], errors="coerce").astype(float)
        valid = (~o.replace([np.inf, -np.inf], np.nan).isna()) & (~p.replace([np.inf, -np.inf], np.nan).isna())
        if not valid.any():
            return None

        o = o[valid]; p = p[valid]

        if (o != 0).any():
            err = (np.abs(p - o) / np.where(o == 0, np.nan, np.abs(o))) * 100.0
            err = err.replace([np.inf, -np.inf], np.nan).dropna()
            return float(err.mean()) if not err.empty else None
        else:
            eps = max(1e-9, float(np.nanmean(np.abs(o))))
            err = (np.abs(p - o) / eps) * 100.0
            return float(np.nanmean(err))


    def check_filling_error(
        self,
        nr_iterations: int,
        data_name: str,
        filling_function: Union[str, Callable[..., Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        **options,
    ) -> None:
        """
        Evaluate filling performance using repeated artificial-gap tests.

        The method repeatedly creates artificial gaps, applies a filling method,
        and computes the average reconstruction error.

        Parameters
        ----------
        nr_iterations : int
            Number of repetitions used to estimate the average error.
        data_name : str
            Name of the column to evaluate.
        filling_function : str or callable
            Filling method to apply. Can be the name of a method or a callable.
        test_data_range : tuple
            Range of data over which the evaluation is performed.
        nr_small_gaps : int, default=0
            Number of small gaps to generate per iteration.
        max_size_small_gaps : int, default=0
            Maximum size of small gaps (in number of samples).
        nr_large_gaps : int, default=0
            Number of large gaps to generate per iteration.
        max_size_large_gaps : int, default=0
            Maximum size of large gaps (in number of samples).
        random_state : int, optional
            Random seed for reproducibility.
        **options
            Additional keyword arguments passed to the filling function.

        Returns
        -------
        None

        Notes
        -----
        - Each iteration creates artificial gaps and evaluates reconstruction error.
        - The average error (in percent) is stored in ``self.filling_error``.
        - The dataset itself is not modified.
        """
        if nr_small_gaps == 0 and nr_large_gaps == 0:
            raise ValueError("Specify nr_small_gaps and/or nr_large_gaps > 0.")

        errors: list[float] = []

        prev_fill_warn = getattr(self, "_filling_warning_issued", False)
        prev_rain_warn = getattr(self, "_rain_warning_issued", False)
        self._filling_warning_issued = True
        self._rain_warning_issued = True

        try:
            for i in range(nr_iterations):
                seed = None if random_state is None else int(random_state + i)
                err = self._calculate_filling_error(
                    data_name,
                    filling_function,
                    test_data_range,
                    nr_small_gaps=nr_small_gaps,
                    max_size_small_gaps=max_size_small_gaps,
                    nr_large_gaps=nr_large_gaps,
                    max_size_large_gaps=max_size_large_gaps,
                    random_state=seed,
                    **options,
                )
                if err is not None and np.isfinite(err):
                    errors.append(float(err))

            if len(errors) == 0:
                raise ValueError(
                    "No valid filling error could be computed. Check `arange`, gap sizes, "
                    "method parameters, and ensure your filling method actually imputes values."
                )

            avg = float(np.mean(errors))

            if not hasattr(self, "filling_error") or not isinstance(self.filling_error, pd.DataFrame):
                self.filling_error = pd.DataFrame(columns=["imputation error [%]"])
            if "imputation error [%]" not in self.filling_error.columns:
                self.filling_error["imputation error [%]"] = np.nan

            self.filling_error.loc[data_name, "imputation error [%]"] = avg
            # print(
            #     "Average deviation of imputed points from the original ones is "
            #     f"{avg:.2f}%. This value is saved in self.filling_error."
            # )
        finally:
            self._filling_warning_issued = prev_fill_warn
            self._rain_warning_issued = prev_rain_warn

            n_attempts = nr_iterations
            n_success = len(errors)
            mean_err = float(np.mean(errors)) if errors else np.nan
            std_err = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
            success_rate = (n_success / n_attempts) if n_attempts > 0 else np.nan

            row = {
                "method": filling_function,
                "mean_error_pct": mean_err,
                "std_error_pct": std_err,
                "n_success": n_success,
                "n_attempts": n_attempts,
                "success_rate": success_rate,
            }
            summary = pd.Series(row)

        return summary



    def compare_filling_methods(
        self,
        data_name: str,
        method_specs: Dict[str, Dict[str, Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_iterations: int = 5,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        return_errors: bool = False,
        plot: bool = False,
    ) -> pd.DataFrame:
        """
        Compare multiple filling methods using artificial-gap tests.

        The method evaluates several filling approaches by repeatedly creating
        artificial gaps, applying each method, and summarizing the resulting
        reconstruction errors.

        Parameters
        ----------
        data_name : str
            Name of the column whose filling performance is evaluated.
        method_specs : dict of str to dict
            Mapping from method labels to specification dictionaries.
            Each specification must include a ``"filling_function"`` entry
            containing either the name of a filling method or a callable.
            Additional entries are passed as keyword arguments to the method.
        test_data_range : tuple
            Range of data over which artificial gaps are created and evaluation
            is performed.
        nr_iterations : int, default=5
            Number of random trials performed for each method.
        nr_small_gaps : int, default=0
            Number of small gaps to generate per iteration.
        max_size_small_gaps : int, default=0
            Maximum size of small gaps (in number of samples).
        nr_large_gaps : int, default=0
            Number of large gaps to generate per iteration.
        max_size_large_gaps : int, default=0
            Maximum size of large gaps (in number of samples).
        random_state : int, optional
            Seed base for reproducibility across methods and iterations.
        return_errors : bool, default=False
            If ``True``, include per-iteration errors in the returned DataFrame.
        plot : bool, default=False
            If ``True``, generate a bar chart of the average filling error for
            each method.

        Returns
        -------
        pandas.DataFrame
            DataFrame summarizing the performance of each filling method. The table
            includes mean error, standard deviation, number of successful runs,
            number of attempts, and success rate. If ``return_errors=True``, the
            per-iteration error values are also included.

        Notes
        -----
        - If a method specification does not provide ``to_fill``, it is set
        automatically to ``data_name``.
        - Artificial gaps are generated independently for each iteration.
        - Results are sorted by mean reconstruction error, with lower values
        indicating better performance.
        - The dataset itself is not modified.
        """
        if nr_small_gaps == 0 and nr_large_gaps == 0:
            raise ValueError(
                "Please specify at least one of nr_small_gaps or nr_large_gaps > 0."
            )

        rows: List[Dict[str, Any]] = []

        for label, spec in method_specs.items():
            if "filling_function" not in spec:
                raise ValueError(f"method_specs['{label}'] must include 'filling_function'.")

            filling_function = spec["filling_function"]

            errors: List[float] = []
            for i in range(nr_iterations):
                rs = (None if random_state is None else (random_state + hash(label) + i) % (2**31 - 1))

                try:
                    err = self._calculate_filling_error(
                        data_name=data_name,
                        filling_function=filling_function,
                        test_data_range=test_data_range,
                        nr_small_gaps=nr_small_gaps,
                        max_size_small_gaps=max_size_small_gaps,
                        nr_large_gaps=nr_large_gaps,
                        max_size_large_gaps=max_size_large_gaps,
                        random_state=rs,
                        **{k: v for k, v in spec.items() if k != "filling_function"},
                    )
                except Exception:
                    
                    err = None

                if err is not None and np.isfinite(err):
                    errors.append(float(err))

            n_attempts = nr_iterations
            n_success = len(errors)
            mean_err = float(np.mean(errors)) if errors else np.nan
            std_err = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
            success_rate = (n_success / n_attempts) if n_attempts > 0 else np.nan

            row = {
                "method": label,
                "mean_error_pct": mean_err,
                "std_error_pct": std_err,
                "n_success": n_success,
                "n_attempts": n_attempts,
                "success_rate": success_rate,
            }
            if return_errors:
                row["errors"] = errors
            rows.append(row)

        summary = pd.DataFrame(rows).sort_values("mean_error_pct", na_position="last").reset_index(drop=True)

        if plot and not summary.empty and summary["mean_error_pct"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(summary))
            ax.bar(x, summary["mean_error_pct"].fillna(0.0))
            ax.set_xticks(x)
            ax.set_xticklabels(summary["method"], rotation=30, ha="right")
            ax.set_ylabel("Mean error (%)")
            ax.set_title(f"Filling method comparison for '{data_name}'")
            if summary["std_error_pct"].notna().any():
                ax.errorbar(
                    x,
                    summary["mean_error_pct"].fillna(0.0),
                    yerr=summary["std_error_pct"].fillna(0.0),
                    fmt="none",
                    capsize=4,
                )
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()

        return summary
