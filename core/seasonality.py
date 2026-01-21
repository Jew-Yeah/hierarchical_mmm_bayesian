# core/stage_02_seasonality.py
"""
Stage 02: Seasonality (weekly) via Wordstat demand
--------------------------------------------------
This stage runs right after preprocessing and works ONLY on weekly/monthly
(because your demand data exists only for weekly & monthly).

We do:
1) Build weekly season factor S_w from demand_weekly:
      S_w = demand_w / mean(demand_train)
   with optional smoothing + clipping for robustness.

2) Deseasonalize leads on weekly level:
      leads_clean_w = leads_w / S_w

3) Optional: aggregate cleaned weekly series to monthly:
      leads_clean_month = sum(leads_clean_week within month)

Important:
- We do NOT "invent" daily demand.
- We do NOT apply seasonality to adstock/resources. Only to leads (target).
- train_mask is supported to avoid leakage in backtests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd


# ----------------------------
# Public helpers
# ----------------------------

def detect_lead_columns_weekly(df_weekly: pd.DataFrame) -> List[str]:
    """
    Detect lead columns in the combined_weekly table.
    Convention (from preprocessing stage): columns end with "__leads".
    """
    return [c for c in df_weekly.columns if c.endswith("__leads")]


def compute_weekly_season_factor(
    df_weekly: pd.DataFrame,
    demand_col: str = "demand_queries",
    week_col: str = "week_start",
    train_mask: Optional[pd.Series] = None,
    smooth_window: int = 0,
    smooth_method: str = "mean",         # "mean" | "median"
    clip_range: Tuple[float, float] = (0.6, 1.6),
    fill_missing: float = 1.0,
) -> pd.Series:
    """
    Compute weekly season factor S_w from demand_col:
      S_w = demand_w / mean(demand_train)

    Parameters
    ----------
    train_mask:
        Boolean series aligned to df_weekly rows. If provided, mean demand is computed only on train rows.
    smooth_window:
        If >=2, demand is smoothed with rolling mean/median (centered).
    clip_range:
        Clip S_w into [low, high] to prevent extreme swings.
    fill_missing:
        If demand or mean cannot be computed, fill season factor with this value.

    Returns
    -------
    pd.Series season_factor aligned to df_weekly index.
    """
    if week_col not in df_weekly.columns:
        raise ValueError(f"Missing week_col='{week_col}' in df_weekly columns")

    if demand_col not in df_weekly.columns:
        # No demand -> no seasonality
        return pd.Series([fill_missing] * len(df_weekly), index=df_weekly.index, name="season_factor")

    demand = pd.to_numeric(df_weekly[demand_col], errors="coerce").astype(float)

    # Optional smoothing (robustness)
    if smooth_window and smooth_window >= 2:
        if smooth_method not in ("mean", "median"):
            raise ValueError("smooth_method must be 'mean' or 'median'")

        roll = demand.rolling(window=smooth_window, center=True, min_periods=max(1, smooth_window // 2))
        demand = roll.mean() if smooth_method == "mean" else roll.median()

    # Define train mask
    if train_mask is None:
        train_mask = pd.Series([True] * len(df_weekly), index=df_weekly.index)
    else:
        train_mask = pd.Series(train_mask, index=df_weekly.index).fillna(False).astype(bool)

    # Mean demand on train rows (ignore NaN, ignore non-positive)
    train_demand = demand.where(train_mask)
    train_demand = train_demand.where(train_demand > 0)

    mean_demand = float(train_demand.mean(skipna=True)) if train_demand.notna().any() else np.nan

    if not np.isfinite(mean_demand) or mean_demand <= 0:
        # Cannot compute mean demand -> disable seasonality
        sf = pd.Series([fill_missing] * len(df_weekly), index=df_weekly.index, name="season_factor")
        return sf

    sf = demand / mean_demand
    sf = sf.replace([np.inf, -np.inf], np.nan)

    # If demand is missing -> default
    sf = sf.fillna(fill_missing)

    # Clip
    low, high = clip_range
    sf = sf.clip(lower=low, upper=high)

    sf.name = "season_factor"
    return sf


def deseasonalize_weekly_leads(
    df_weekly: pd.DataFrame,
    lead_cols: Optional[Iterable[str]] = None,
    season_factor: Optional[pd.Series] = None,
    season_col: str = "season_factor",
    suffix_clean: str = "__leads_clean",
) -> pd.DataFrame:
    """
    Create cleaned lead columns:
      leads_clean = leads / season_factor

    Output adds new columns per channel. Does NOT overwrite originals.
    """
    df = df_weekly.copy()

    if season_factor is not None:
        df[season_col] = pd.Series(season_factor, index=df.index).astype(float)
    elif season_col not in df.columns:
        df[season_col] = 1.0

    sf = pd.to_numeric(df[season_col], errors="coerce").astype(float).replace([0.0, np.inf, -np.inf], np.nan)
    sf = sf.fillna(1.0)

    if lead_cols is None:
        lead_cols = detect_lead_columns_weekly(df)

    lead_cols = list(lead_cols)
    if not lead_cols:
        # Nothing to clean
        return df

    for col in lead_cols:
        y = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0.0)
        clean_col = col.replace("__leads", suffix_clean) if col.endswith("__leads") else f"{col}_clean"
        df[clean_col] = y / sf

    return df


def aggregate_weekly_to_monthly_clean(
    df_weekly: pd.DataFrame,
    week_col: str = "week_start",
    cols_sum: Optional[Iterable[str]] = None,
    include_season_stats: bool = True,
    season_col: str = "season_factor",
) -> pd.DataFrame:
    """
    Aggregate weekly dataframe to monthly. Typically used on cleaned lead cols.

    - month_start = first day of month of week_start
    - sums cols_sum by month
    - optionally keeps some seasonality diagnostics

    Note:
    The most correct way to go "monthly in original scale" is:
      leads_month = sum_w (season_w * leads_clean_w)
    This function focuses on producing monthly CLEAN sums. Use reseasonalize_monthly()
    if you need original scale back.
    """
    if week_col not in df_weekly.columns:
        raise ValueError(f"Missing week_col='{week_col}'")

    df = df_weekly.copy()
    df[week_col] = pd.to_datetime(df[week_col], errors="raise")
    df["month_start"] = df[week_col].dt.to_period("M").dt.to_timestamp()

    if cols_sum is None:
        # default: sum all cleaned lead columns + spends if you want
        cols_sum = [c for c in df.columns if c.endswith("__leads_clean")]

    cols_sum = list(cols_sum)

    agg = {c: "sum" for c in cols_sum}
    if include_season_stats and season_col in df.columns:
        agg.update({
            season_col: "mean",
            f"{season_col}__min": (season_col, "min"),
            f"{season_col}__max": (season_col, "max"),
        })
        # pandas named aggregation form
        monthly = df.groupby("month_start").agg(**{
            **{c: (c, "sum") for c in cols_sum},
            season_col: (season_col, "mean"),
            f"{season_col}__min": (season_col, "min"),
            f"{season_col}__max": (season_col, "max"),
        }).reset_index()
    else:
        monthly = df.groupby("month_start")[cols_sum].sum().reset_index()

    return monthly


def reseasonalize_weekly_predictions(
    pred_clean: Union[pd.Series, np.ndarray],
    season_factor: Union[pd.Series, np.ndarray],
) -> np.ndarray:
    """
    Convert cleaned weekly predictions back to original scale:
      pred = season_factor * pred_clean
    """
    pc = np.asarray(pred_clean, dtype=float)
    sf = np.asarray(season_factor, dtype=float)
    sf = np.where(np.isfinite(sf) & (sf > 0), sf, 1.0)
    return sf * pc


def reseasonalize_monthly_from_weekly(
    weekly_pred_clean: Union[pd.Series, np.ndarray],
    weekly_season_factor: Union[pd.Series, np.ndarray],
    weekly_week_start: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """
    Correct monthly back-transform from weekly cleaned predictions:
      month_pred = sum_w (S_w * pred_clean_w)

    Returns DataFrame: month_start, pred
    """
    df = pd.DataFrame({
        "week_start": pd.to_datetime(pd.Series(weekly_week_start), errors="raise"),
        "pred_clean": np.asarray(weekly_pred_clean, dtype=float),
        "season_factor": np.asarray(weekly_season_factor, dtype=float),
    })
    df["season_factor"] = df["season_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df.loc[df["season_factor"] <= 0, "season_factor"] = 1.0

    df["pred"] = df["pred_clean"] * df["season_factor"]
    df["month_start"] = df["week_start"].dt.to_period("M").dt.to_timestamp()

    out = df.groupby("month_start", as_index=False)["pred"].sum()
    return out


# ----------------------------
# Stage runner: plug into pipeline
# ----------------------------

@dataclass(frozen=True)
class SeasonalityResult:
    weekly: pd.DataFrame
    season_factor_col: str
    cleaned_suffix: str


def stage_apply_weekly_seasonality(
    combined_weekly: pd.DataFrame,
    demand_col: str = "demand_queries",
    week_col: str = "week_start",
    lead_cols: Optional[Iterable[str]] = None,
    train_mask: Optional[pd.Series] = None,
    smooth_window: int = 0,
    smooth_method: str = "mean",
    clip_range: Tuple[float, float] = (0.6, 1.6),
    fill_missing: float = 1.0,
    season_col: str = "season_factor",
    cleaned_suffix: str = "__leads_clean",
) -> SeasonalityResult:
    """
    Main stage entry:
    - compute season factor S_w from demand
    - create cleaned leads columns
    Returns SeasonalityResult with updated weekly df.
    """
    sf = compute_weekly_season_factor(
        df_weekly=combined_weekly,
        demand_col=demand_col,
        week_col=week_col,
        train_mask=train_mask,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        clip_range=clip_range,
        fill_missing=fill_missing,
    )

    weekly = deseasonalize_weekly_leads(
        df_weekly=combined_weekly,
        lead_cols=lead_cols,
        season_factor=sf,
        season_col=season_col,
        suffix_clean=cleaned_suffix,
    )

    return SeasonalityResult(
        weekly=weekly,
        season_factor_col=season_col,
        cleaned_suffix=cleaned_suffix,
    )
