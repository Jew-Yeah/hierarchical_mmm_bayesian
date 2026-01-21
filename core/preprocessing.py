# 01_stage_preprocessing.py
"""
Stage 01: Preprocessing / data loading

Goal:
- By dataset name (e.g. "dataset_caravan_mvp") read all input CSVs
- Return everything in a single dictionary (plus an optional in-module cache)

Expected folder structure (your current one):
data/
  dataset_caravan_mvp/
    flow_weekly/
      StatisticAdvertising - Context_weekly.csv
      ...
    stock_weekly/
      StatisticAdvertising - Car_weekly.csv
      ...
    demand_weekly/
      wordstat_dynamic.csv

Notes:
- flow_weekly / stock_weekly CSVs are expected to have columns:
  week_id, week_start, week_end, spend_rub, leads_count
- wordstat_dynamic.csv is expected to be semicolon-separated with BOM and columns:
  "Неделя с", "Число запросов", "Доля от всех запросов, %", ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

import pandas as pd


# ----------------------------
# Public API
# ----------------------------

_DATASET_LOADERS = {}
_CACHE: Dict[str, Dict[str, Any]] = {}


def load_dataset(
    dataset_name: str = "dataset_caravan_mvp",
    data_root: Union[str, Path] = "data",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Main entry point:
    - dataset_name: e.g. "dataset_caravan_mvp"
    - data_root: root folder that contains dataset folders (default: ./data)
    Returns a dictionary with parsed tables and merged weekly dataframe.
    """
    key = f"{Path(data_root).resolve()}::{dataset_name}"
    if use_cache and key in _CACHE:
        return _CACHE[key]

    loader = _DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. "
            f"Known: {sorted(_DATASET_LOADERS.keys())}"
        )

    out = loader(data_root=Path(data_root), dataset_name=dataset_name)
    if use_cache:
        _CACHE[key] = out
    return out


def build_stock_channel_configs(
    stock_constants: pd.DataFrame,
    present_stock_channels: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Convert constants table -> per-channel config dict.

    Returned dict keys are channel names (e.g. "car"). Values are dicts accepted by
    core.fit_stock_channels.stage_fit_stock_channels().

    Missing fields are left as None so CLI can fill them in.
    """

    # defaults (MVP)
    defaults = {
        "cost_per_unit": None,
        "life_weeks": 26,
        "unit_decay": 0.995,
        "avg_units_target": None,
        "max_units": None,
        "calibration_mode": "monthly_mean",
        "initial_active_units": None,
    }

    cfg: Dict[str, Dict[str, Any]] = {ch: dict(defaults) for ch in present_stock_channels}

    if stock_constants is None or stock_constants.empty:
        return cfg

    df = stock_constants.copy()
    if "channel" not in df.columns:
        return cfg

    df["channel"] = df["channel"].astype(str).str.strip().str.lower()

    for ch in present_stock_channels:
        row = df[df["channel"] == str(ch).lower()]
        if row.empty:
            continue
        r = row.iloc[0].to_dict()

        def _get_float(name: str) -> Optional[float]:
            v = r.get(name, None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return float(v)
            except Exception:
                return None

        def _get_int(name: str, default: int) -> int:
            v = r.get(name, None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return int(default)
            try:
                return int(float(v))
            except Exception:
                return int(default)

        cfg[ch]["cost_per_unit"] = _get_float("cost_per_unit")
        cfg[ch]["life_weeks"] = _get_int("life_weeks", defaults["life_weeks"])
        ud = _get_float("unit_decay")
        cfg[ch]["unit_decay"] = float(ud) if ud is not None else defaults["unit_decay"]
        cfg[ch]["avg_units_target"] = _get_float("avg_units_target")
        cfg[ch]["max_units"] = _get_float("max_units")
        cm = r.get("calibration_mode", defaults["calibration_mode"])
        cfg[ch]["calibration_mode"] = str(cm).strip() if cm is not None else defaults["calibration_mode"]
        cfg[ch]["initial_active_units"] = _get_float("initial_active_units")

    return cfg


def register_dataset_loader(name: str):
    """Decorator to register a dataset loader by name."""
    def _wrap(fn):
        _DATASET_LOADERS[name] = fn
        return fn
    return _wrap


# ----------------------------
# Dataset: dataset_caravan_mvp
# ----------------------------

@register_dataset_loader("dataset_caravan_mvp")
def _load_dataset_caravan_mvp(data_root: Path, dataset_name: str) -> Dict[str, Any]:
    dataset_dir = (data_root / dataset_name).resolve()
    _assert_exists(dataset_dir, f"Dataset folder not found: {dataset_dir}")

    flow_dir = dataset_dir / "flow_weekly"
    stock_dir = dataset_dir / "stock_weekly"
    demand_dir = dataset_dir / "demand_weekly"

    _assert_exists(flow_dir, f"Missing folder: {flow_dir}")
    _assert_exists(stock_dir, f"Missing folder: {stock_dir}")
    _assert_exists(demand_dir, f"Missing folder: {demand_dir}")

    flow_weekly = _load_weekly_channel_folder(flow_dir, kind="flow")
    stock_weekly = _load_weekly_channel_folder(stock_dir, kind="stock")
    demand_weekly = _load_wordstat_weekly(demand_dir / "wordstat_dynamic.csv")

    # Optional constants (variant 1): data/<dataset>/constants/stock_channels.csv
    constants_dir = dataset_dir / "constants"
    stock_constants_path = constants_dir / "stock_channels.csv"
    if stock_constants_path.exists():
        stock_constants = _load_stock_constants(stock_constants_path)
    else:
        stock_constants = pd.DataFrame()

    combined_weekly = _build_combined_weekly(flow_weekly, stock_weekly, demand_weekly)

    # basic integrity checks (hard fail early)
    _validate_weekly_index(combined_weekly)

    return {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "flow_weekly": flow_weekly,     # dict[channel -> df]
        "stock_weekly": stock_weekly,   # dict[channel -> df]
        "present_stock_channels": sorted(list(stock_weekly.keys())),
        "stock_constants": stock_constants,
        "demand_weekly": demand_weekly, # df
        "combined_weekly": combined_weekly,  # df
        "weeks": combined_weekly["week_start"].tolist(),
    }


# ----------------------------
# Helpers: reading files
# ----------------------------

def _load_weekly_channel_folder(folder: Path, kind: str) -> Dict[str, pd.DataFrame]:
    """
    Reads all *.csv in folder as separate channels.
    Returns dict: channel_name -> weekly df.
    """
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {folder}")

    out: Dict[str, pd.DataFrame] = {}
    for fp in files:
        channel = _channel_name_from_filename(fp.name, kind=kind)
        df = _read_weekly_channel_csv(fp)
        out[channel] = df
    return out


def _read_weekly_channel_csv(path: Path) -> pd.DataFrame:
    """
    Expected columns:
      week_id, week_start, week_end, spend_rub, leads_count
    """
    df = pd.read_csv(path)

    required = {"week_start", "spend_rub", "leads_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}. Got: {list(df.columns)}")

    # Parse dates
    df["week_start"] = pd.to_datetime(df["week_start"], errors="raise")
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")

    # Numeric columns
    df["spend_rub"] = pd.to_numeric(df["spend_rub"], errors="coerce").fillna(0.0)
    df["leads_count"] = pd.to_numeric(df["leads_count"], errors="coerce").fillna(0.0)

    # Sort + drop duplicates by week_start (keep first)
    df = df.sort_values("week_start").drop_duplicates(subset=["week_start"], keep="first").reset_index(drop=True)

    # Minimal sanity
    if (df["spend_rub"] < 0).any():
        raise ValueError(f"{path.name}: found negative spend_rub")
    if (df["leads_count"] < 0).any():
        raise ValueError(f"{path.name}: found negative leads_count")

    # Standard output cols
    keep_cols = [c for c in ["week_id", "week_start", "week_end", "spend_rub", "leads_count"] if c in df.columns]
    return df[keep_cols].copy()


def _load_wordstat_weekly(path: Path) -> pd.DataFrame:
    """
    Parses wordstat_dynamic.csv (weekly) with typical format:
      'Неделя с;Число запросов;Доля от всех запросов, %;...'
      12.05.2025;1 231;0,000047;
    We return:
      week_start (datetime), demand_queries (float), demand_share (float)
    """
    _assert_exists(path, f"Missing demand file: {path}")

    # sep=';' and BOM-safe encoding
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # Identify columns
    # Common: "Неделя с", "Число запросов", "Доля от всех запросов, %"
    col_week = _find_first_existing(df.columns, ["Неделя с", "week_start", "Week start", "Date"])
    col_cnt = _find_first_existing(df.columns, ["Число запросов", "queries", "demand", "Count"])
    col_share = _find_first_existing(df.columns, ["Доля от всех запросов, %", "share", "Share"])

    if col_week is None or col_cnt is None:
        raise ValueError(
            f"{path.name}: cannot find required columns. "
            f"Got columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["week_start"] = pd.to_datetime(df[col_week].astype(str).str.strip(), format="%d.%m.%Y", errors="raise")

    # numbers: "1 231" -> 1231 ; "0,000047" -> 0.000047
    out["demand_queries"] = (
        df[col_cnt]
        .astype(str)
        .str.replace("\u00a0", " ", regex=False)   # non-breaking space
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["demand_queries"] = pd.to_numeric(out["demand_queries"], errors="coerce")

    if col_share is not None:
        out["demand_share"] = (
            df[col_share]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        out["demand_share"] = pd.to_numeric(out["demand_share"], errors="coerce")
    else:
        out["demand_share"] = pd.NA

    out = out.sort_values("week_start").drop_duplicates(subset=["week_start"], keep="first").reset_index(drop=True)
    return out


def _load_stock_constants(path: Path) -> pd.DataFrame:
    """Load stock channel constants table (variant 1).

    Expected columns (any subset is ok):
      channel,cost_per_unit,life_weeks,unit_decay,avg_units_target,max_units,calibration_mode,initial_active_units

    Values are kept as raw types; conversion happens in build_stock_channel_configs().
    """
    df = pd.read_csv(path)
    # normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    if "channel" in df.columns:
        df["channel"] = df["channel"].astype(str).str.strip().str.lower()
    return df


# ----------------------------
# Helpers: merge + validation
# ----------------------------

def _build_combined_weekly(
    flow_weekly: Dict[str, pd.DataFrame],
    stock_weekly: Dict[str, pd.DataFrame],
    demand_weekly: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns a single weekly dataframe with columns:
      week_start,
      flow__<channel>__spend, flow__<channel>__leads,
      stock__<channel>__spend, stock__<channel>__leads,
      demand_queries, demand_share
    """
    # Start from the union of week_start across all sources
    weeks = None

    def _collect_weeks(df: pd.DataFrame):
        return set(df["week_start"].tolist())

    for d in list(flow_weekly.values()) + list(stock_weekly.values()) + [demand_weekly]:
        w = _collect_weeks(d)
        weeks = w if weeks is None else (weeks | w)

    combined = pd.DataFrame({"week_start": sorted(weeks)})

    # merge each channel
    for ch, df in flow_weekly.items():
        tmp = df[["week_start", "spend_rub", "leads_count"]].copy()
        tmp = tmp.rename(columns={
            "spend_rub": f"flow__{ch}__spend",
            "leads_count": f"flow__{ch}__leads",
        })
        combined = combined.merge(tmp, on="week_start", how="left")

    for ch, df in stock_weekly.items():
        tmp = df[["week_start", "spend_rub", "leads_count"]].copy()
        tmp = tmp.rename(columns={
            "spend_rub": f"stock__{ch}__spend",
            "leads_count": f"stock__{ch}__leads",
        })
        combined = combined.merge(tmp, on="week_start", how="left")

    # merge demand
    dcols = [c for c in ["week_start", "demand_queries", "demand_share"] if c in demand_weekly.columns]
    combined = combined.merge(demand_weekly[dcols], on="week_start", how="left")

    # Fill missing spends/leads with zeros; demand can be missing => keep NaN (or fill later)
    for c in combined.columns:
        if c.endswith("__spend") or c.endswith("__leads"):
            combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0.0)

    combined = combined.sort_values("week_start").reset_index(drop=True)
    return combined


def _validate_weekly_index(df: pd.DataFrame) -> None:
    if "week_start" not in df.columns:
        raise ValueError("combined_weekly missing week_start")

    if df["week_start"].isna().any():
        raise ValueError("combined_weekly has NaN week_start")

    if not df["week_start"].is_unique:
        raise ValueError("combined_weekly has duplicate week_start")

    # Ensure 7-day step (no gaps) if there are at least 2 weeks
    if len(df) >= 2:
        diffs = df["week_start"].sort_values().diff().dropna().dt.days
        if not (diffs == 7).all():
            bad = diffs[diffs != 7]
            raise ValueError(f"Non-weekly continuity detected. Bad diffs (days): {bad.tolist()}")


# ----------------------------
# Utilities
# ----------------------------

def _channel_name_from_filename(filename: str, kind: str) -> str:
    """
    Makes a stable short channel name from a file name.
    Examples:
      "StatisticAdvertising - Context_weekly.csv" -> "context"
      "StatisticAdvertising - Car_weekly.csv" -> "car"
    """
    name = filename
    name = name.replace(".csv", "")
    name = name.replace("_weekly", "")
    name = name.replace("StatisticAdvertising - ", "")
    name = name.strip()

    # slugify-ish
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in [" ", "-", "_"]:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")

    # add prefix for safety if needed
    if not slug:
        slug = f"{kind}_channel"
    return slug


def _find_first_existing(cols, candidates):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def _assert_exists(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(msg)
