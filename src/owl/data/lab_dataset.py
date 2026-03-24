# -*- coding: utf-8 -*-
"""
LabDataset abstraction for handling lab experiments data in urban water systems.

This module defines the 'LabDataset' class used to handle lab experiments measurements in the OpenWaterLab (OWL) ecosystem.

The LabDataset class extends the Dataset class by providing additional functionalities for using lab data for validation purposes.

Classes
-------
LabDataset
    Extends Dataset class for lab experiments data measurements.
"""
from __future__ import annotations
import sys

import matplotlib.pyplot as plt  
import warnings as wn
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, List, Literal, Union

from .dataset import Dataset

@dataclass
class ValidationMetrics:
    n: int
    mae: float
    rmse: float
    mbe: float            
    mape: float
    r: float              
    r2: float
    ccc: float            
    slope: float
    intercept: float

class LabDataset(Dataset):
    """
    Dataset class for handling laboratory and experimental data.

    This class extends the base ``Dataset`` with functionality specific to
    laboratory measurements, including support for metadata, sample identifiers,
    and quality control annotations.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing laboratory measurements.
    timedata_column : str, default="index"
        Column used as the time reference, or ``"index"`` if the index
        represents time.
    data_type : str, default="WWTP"
        Type of data represented in the dataset.
    experiment_tag : str, default="No tag given"
        Label identifying the experiment or dataset.
    time_unit : str, optional
        Unit of time used for the dataset (e.g., ``"s"``, ``"min"``, ``"h"``).
    units : dict, optional
        Mapping of column names to their corresponding units.
    meta : pandas.DataFrame, optional
        Metadata associated with each data point, aligned with the dataset index.
    id_column : str, optional
        Name of the column containing sample or experiment identifiers.

    Attributes
    ----------
    meta : pandas.DataFrame
        Metadata associated with the dataset.
    id_column : str or None
        Column used to identify samples or experiments, if available.
    meta_qc : pandas.DataFrame
        DataFrame storing quality control information for the dataset.

    Notes
    -----
    - If ``meta`` is not provided, an empty DataFrame is initialized with the
      same index as the data.
    - The ``id_column`` is only assigned if it exists in the dataset.
    - The class is designed for laboratory workflows where additional metadata
      and quality control tracking are required.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        timedata_column: str = "index",
        data_type: str = "WWTP",
        experiment_tag: str = "No tag given",
        time_unit: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
        meta: Optional[pd.DataFrame] = None,       
        id_column: Optional[str] = None,           
    ):
        super().__init__(
            data=data,
            timedata_column=timedata_column,
            data_type=data_type,
            experiment_tag=experiment_tag,
            time_unit=time_unit,
            units=({} if units is None else units),
        )
        self.meta: pd.DataFrame = (
            meta.copy() if isinstance(meta, pd.DataFrame) else pd.DataFrame(index=self.data.index)
        )
        self.id_column: Optional[str] = id_column if id_column in self.data.columns else None

        self.meta_qc: pd.DataFrame = pd.DataFrame(index=self.data.index)


    def apply_dilutions_and_units(
        self,
        analytes: Iterable[str],
        *,
        dilution_factors: Union[float, Dict[str, float]] = 1.0,
        unit_map: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> None:
        """
        Apply dilution factors and unit conversions to selected analytes.

        The method adjusts measured values based on specified dilution factors
        and optionally converts them to target units using provided conversion
        factors.

        Parameters
        ----------
        analytes : iterable of str
            Names of the columns (analytes) to process.
        dilution_factors : float or dict of str to float, default=1.0
            Dilution factors applied to the selected analytes. If a single value is
            provided, it is applied to all analytes. If a dictionary is provided,
            factors are applied per analyte.
        unit_map : dict of str to (str, float), optional
            Mapping of analyte names to target units and conversion factors.
            Each entry should be of the form ``{column: (unit, factor)}``, where
            ``factor`` is multiplied with the data to convert units.

        Returns
        -------
        None

        Notes
        -----
        - Dilution factors are applied before unit conversions.
        - Unit conversions are performed by multiplying values with the provided
        conversion factor.
        - The dataset is modified in place.
        """
        for a in analytes:
            factor = dilution_factors if isinstance(dilution_factors, (int, float)) else dilution_factors.get(a, 1.0)
            self.data[a] = pd.to_numeric(self.data[a], errors="coerce") * factor
            if unit_map and a in unit_map:
                new_unit, u_factor = unit_map[a]
                self.data[a] = self.data[a] * u_factor
                self.units[a] = new_unit

    def control_chart(
        self,
        analyte: str,
        *,
        subgroup: Optional[str] = None,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Compute Shewhart X-bar control chart statistics.

        The method calculates the mean and control limits (±3 standard deviations)
        either over a rolling window or within defined subgroups.

        Parameters
        ----------
        analyte : str
            Name of the column to analyze.
        subgroup : str, optional
            Column used to define subgroups (e.g., batches). If provided,
            statistics are computed per subgroup instead of using a rolling window.
        window : int, default=20
            Size of the rolling window used when ``subgroup`` is not provided.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the computed mean and control limits, suitable
            for plotting a control chart.

        Notes
        -----
        - Control limits are defined as ``mean ± 3 * standard deviation``.
        - If ``subgroup`` is provided, statistics are computed within each subgroup.
        - The method is intended for quality control and monitoring applications.
        """
        s = pd.to_numeric(self.data[analyte], errors="coerce")
        if subgroup and subgroup in self.meta.columns:
            grp = self.meta.groupby(self.meta[subgroup])
            mu = grp.apply(lambda g: s.reindex(g.index).mean())
            sd = grp.apply(lambda g: s.reindex(g.index).std(ddof=1))
            out = pd.DataFrame({"center": mu, "ucl": mu + 3 * sd, "lcl": mu - 3 * sd})
        else:
            mu = s.rolling(window, min_periods=max(5, window//5)).mean()
            sd = s.rolling(window, min_periods=max(5, window//5)).std(ddof=1)
            out = pd.DataFrame({"center": mu, "ucl": mu + 3 * sd, "lcl": mu - 3 * sd})
        return out
    
    
    def align_with_reference(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        method: Literal["nearest", "linear", "ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
    ) -> pd.DataFrame:
        """
        Align laboratory measurements with a reference time series.

        The method matches discrete laboratory measurements to a continuous
        reference (e.g., sensor or model data) based on timestamps.

        Parameters
        ----------
        analyte : str
            Name of the laboratory data column to align.
        reference : pandas.Series or pandas.DataFrame
            Reference time series to align with the laboratory data.
        method : {"nearest", "linear", "ffill"}, default="nearest"
            Method used to align or interpolate the reference values to the
            laboratory timestamps.
        tolerance : pandas.Timedelta, optional
            Maximum allowed time difference for matching values when using
            ``method="nearest"``. If exceeded, no match is assigned.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by laboratory timestamps with columns ``["lab", "ref"]``.

        Notes
        -----
        - The laboratory data is assumed to be discrete, while the reference is
        continuous or higher-frequency.
        - The alignment is performed on the laboratory timestamps.
        - The chosen method determines how reference values are interpolated or matched.
        """
        if analyte not in self.data.columns:
            raise KeyError(f"'{analyte}' not in LabData.data columns")
        ref = _ensure_series(reference).copy()
        lab = pd.to_numeric(self.data[analyte], errors="coerce").dropna()
        if lab.empty:
            wn.warn("No lab samples to align.", stacklevel=2)
            return pd.DataFrame(columns=["lab","ref"])

        if method == "nearest":
            aligned = ref.reindex(lab.index, method="nearest", tolerance=tolerance)
        elif method == "linear":
            aligned = ref.reindex(ref.index.union(lab.index)).interpolate("time").reindex(lab.index)
        else:  # ffill
            aligned = ref.reindex(ref.index.union(lab.index)).ffill().reindex(lab.index)

        out = pd.DataFrame({"lab": lab, "ref": aligned})
        return out.dropna()
    

    def compute_validation_metrics(
        self,
        pairs: pd.DataFrame,
        *,
        bias_as: Literal["ref_minus_lab","lab_minus_ref"] = "ref_minus_lab",
        zero_guard: float = 1e-12,
    ) -> ValidationMetrics:
        """
        Compute validation metrics between laboratory and reference data.

        The method evaluates common statistical metrics based on paired values
        of laboratory measurements and reference data.

        Parameters
        ----------
        pairs : pandas.DataFrame
            DataFrame containing paired values with columns ``["lab", "ref"]``.
        bias_as : {"ref_minus_lab", "lab_minus_ref"}, default="ref_minus_lab"
            Definition of bias used in the calculations.
        zero_guard : float, default=1e-12
            Small value used to avoid division by zero in metric computations.

        Returns
        -------
        ValidationMetrics
            Object containing the computed validation metrics.

        Notes
        -----
        - Metrics may include bias, RMSE, MAE, and related statistics.
        - The definition of bias depends on the ``bias_as`` parameter.
        - The method assumes that the input data is already aligned.
        """
        
        
        df = pairs.dropna()
        n = len(df)
        if n == 0:
            return ValidationMetrics(0, *[np.nan]*10)

        lab = df["lab"].astype(float)
        ref = df["ref"].astype(float)

        err = (ref - lab) if bias_as == "ref_minus_lab" else (lab - ref)
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mbe  = float(np.mean(err))
        denom = np.maximum(np.abs(ref), zero_guard)
        mape = float(np.mean(np.abs(err) / denom) * 100.0)

        r = float(np.corrcoef(ref, lab)[0,1]) if n > 1 else np.nan
        r2 = r*r if np.isfinite(r) else np.nan
        x = ref.values
        y = lab.values
        x_mean, y_mean = np.mean(x), np.mean(y)
        sxy = np.sum((x - x_mean)*(y - y_mean))
        sxx = np.sum((x - x_mean)**2)
        slope = float(sxy / sxx) if sxx > 0 else np.nan
        intercept = float(y_mean - slope*x_mean) if np.isfinite(slope) else np.nan

        sx = np.var(x, ddof=1)
        sy = np.var(y, ddof=1)
        if n > 1 and np.isfinite(r) and sx > 0 and sy > 0:
            ccc = float(2*r*np.sqrt(sx*sy) / (sx + sy + (x_mean - y_mean)**2))
        else:
            ccc = np.nan

        return ValidationMetrics(n, mae, rmse, mbe, mape, r, r2, ccc, slope, intercept)
    

    def validate_against(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
        bias_as: Literal["ref_minus_lab","lab_minus_ref"] = "ref_minus_lab",
        return_pairs: bool = False,
    ) -> Union[ValidationMetrics, Tuple[ValidationMetrics, pd.DataFrame]]:
        """
        Validate laboratory data against a reference time series.

        The method aligns laboratory measurements with a reference (e.g., sensor
        or model data) and computes validation metrics based on the aligned pairs.

        Parameters
        ----------
        analyte : str
            Name of the laboratory data column to validate.
        reference : pandas.Series or pandas.DataFrame
            Reference time series used for validation.
        align_method : {"nearest", "linear", "ffill"}, default="nearest"
            Method used to align or interpolate the reference values to the
            laboratory timestamps.
        tolerance : pandas.Timedelta, optional
            Maximum allowed time difference for matching values when using
            ``align_method="nearest"``.
        bias_as : {"ref_minus_lab", "lab_minus_ref"}, default="ref_minus_lab"
            Definition of bias used in the validation metrics.
        return_pairs : bool, default=False
            If ``True``, also return the aligned data pairs used for validation.

        Returns
        -------
        ValidationMetrics or (ValidationMetrics, pandas.DataFrame)
            Computed validation metrics. If ``return_pairs=True``, also returns
            the DataFrame of aligned pairs with columns ``["lab", "ref"]``.

        Notes
        -----
        - Alignment is performed using the specified method before computing metrics.
        - The method combines alignment and metric computation into a single workflow.
        - The dataset itself is not modified.
        """
        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        metrics = self.compute_validation_metrics(pairs, bias_as=bias_as)
        return (metrics, pairs) if return_pairs else metrics
    
    def fit_calibration(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
        zero_intercept: bool = False,
    ) -> Dict[str, float]:
        """
        Fit a linear calibration model between laboratory and reference data.

        The method aligns laboratory measurements with a reference time series
        and fits a linear model of the form ``lab = a * ref + b``.

        Parameters
        ----------
        analyte : str
            Name of the laboratory data column to calibrate.
        reference : pandas.Series or pandas.DataFrame
            Reference time series used for calibration.
        align_method : {"nearest", "linear", "ffill"}, default="nearest"
            Method used to align or interpolate the reference values to the
            laboratory timestamps.
        tolerance : pandas.Timedelta, optional
            Maximum allowed time difference for matching values when using
            ``align_method="nearest"``.
        zero_intercept : bool, default=False
            If ``True``, fit the model with zero intercept (``b = 0``).

        Returns
        -------
        dict of str to float
            Dictionary containing the fitted parameters with keys
            ``{"slope", "intercept", "r2"}``.

        Notes
        -----
        - Alignment is performed prior to fitting the calibration model.
        - The model is fitted using paired values of laboratory and reference data.
        - The coefficient of determination (R²) is included as a measure of fit quality.
        - The dataset itself is not modified.
        """

        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        if pairs.empty:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}

        x = pairs["ref"].values
        y = pairs["lab"].values
        if zero_intercept:
            sxx = np.sum(x*x)
            sxy = np.sum(x*y)
            slope = float(sxy / sxx) if sxx > 0 else np.nan
            intercept = 0.0
            yhat = slope * x
        else:
            x_mean, y_mean = np.mean(x), np.mean(y)
            sxx = np.sum((x - x_mean)**2)
            sxy = np.sum((x - x_mean)*(y - y_mean))
            slope = float(sxy / sxx) if sxx > 0 else np.nan
            intercept = float(y_mean - slope*x_mean) if np.isfinite(slope) else np.nan
            yhat = slope * x + intercept

        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

        return {"slope": slope, "intercept": intercept, "r2": r2}

    def apply_calibration(
        self,
        reference: Union[pd.Series, pd.DataFrame],
        slope: float,
        intercept: float = 0.0,
        *,
        name: Optional[str] = None,
    ) -> pd.Series:
        """
        Apply a linear calibration model to reference data.

        The method computes calibrated values using a linear relationship of the form
        ``lab = slope * reference + intercept``.

        Parameters
        ----------
        reference : pandas.Series or pandas.DataFrame
            Reference time series used to generate calibrated values.
        slope : float
            Slope of the calibration model.
        intercept : float, default=0.0
            Intercept of the calibration model.
        name : str, optional
            Name assigned to the resulting calibrated series.

        Returns
        -------
        pandas.Series
            Calibrated values indexed by the reference timestamps.

        Notes
        -----
        - The calibration is applied directly to the reference data.
        - The output is aligned with the reference index.
        - The dataset itself is not modified.
        """
        ref = _ensure_series(reference).astype(float)
        est = slope * ref + intercept
        est.name = name or f"calibrated_{_name_or_default(ref,'ref')}"
        return est

    def rolling_validation(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        window: int = 20,
        step: int = 5,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
    ) -> pd.DataFrame:
        """
        Perform rolling validation between laboratory and reference data.

        The method evaluates validation metrics over successive windows to monitor
        temporal changes or drift in the relationship between laboratory and
        reference measurements.

        Parameters
        ----------
        analyte : str
            Name of the laboratory data column to validate.
        reference : pandas.Series or pandas.DataFrame
            Reference time series used for validation.
        window : int, default=20
            Number of samples in each rolling window.
        step : int, default=5
            Step size between consecutive windows.
        align_method : {"nearest", "linear", "ffill"}, default="nearest"
            Method used to align or interpolate the reference values to the
            laboratory timestamps.
        tolerance : pandas.Timedelta, optional
            Maximum allowed time difference for matching values when using
            ``align_method="nearest"``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by window end points containing validation metrics
            for each window.

        Notes
        -----
        - Alignment is performed prior to computing validation metrics in each window.
        - The method is useful for detecting drift or time-varying performance.
        - The dataset itself is not modified.
        """
        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        if pairs.empty:
            return pd.DataFrame(columns=["n","mae","rmse","mbe","mape","r","r2","ccc","slope","intercept"])

        rows = []
        idxs = pairs.index
        for start in range(0, len(pairs)-window+1, step):
            win = pairs.iloc[start:start+window]
            m = self.compute_validation_metrics(win)
            rows.append({
                "end": idxs[min(start+window-1, len(pairs)-1)],
                **m.__dict__
            })
        out = pd.DataFrame(rows).set_index("end")
        return out


##############################
# Help functions
##############################

def _ensure_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    return x.squeeze() if isinstance(x, pd.DataFrame) else x

def _name_or_default(s: pd.Series, default: str) -> str:
    return s.name if s.name is not None else default