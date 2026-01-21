# core/context_builder.py
"""Context builder.

Pipelines expect a single dict ("ctx") that contains:
- raw loaded dataset tables
- df_weekly: merged weekly table with season_factor and *_leads_clean columns
- stock_channel_configs: per-stock-channel constants (cost_per_unit, etc.)

This module MUST NOT prompt the user. Any console I/O belongs to main.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.preprocessing import load_dataset, build_stock_channel_configs
from core.seasonality import stage_apply_weekly_seasonality, detect_lead_columns_weekly


def build_context(
    dataset_name: str,
    data_root: str = "data",
    *,
    smooth_window: int = 3,
    smooth_method: str = "median",
    clip_range: Tuple[float, float] = (0.6, 1.6),
) -> Dict[str, Any]:
    """Load dataset, apply seasonality, attach constants/configs."""

    raw = load_dataset(dataset_name=dataset_name, data_root=data_root)
    df_weekly = raw["combined_weekly"].copy()

    lead_cols = detect_lead_columns_weekly(df_weekly)
    season_res = stage_apply_weekly_seasonality(
        combined_weekly=df_weekly,
        demand_col="demand_queries",
        week_col="week_start",
        lead_cols=lead_cols,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        clip_range=clip_range,
        fill_missing=1.0,
        season_col="season_factor",
        cleaned_suffix="__leads_clean",
    )

    # constants/configs
    stock_constants = raw.get("stock_constants", pd.DataFrame())
    stock_channel_configs = build_stock_channel_configs(
        stock_constants=stock_constants,
        present_stock_channels=raw.get("present_stock_channels", []),
    )

    ctx: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "data_root": data_root,
        "raw": raw,
        "df_weekly": season_res.weekly,
        "season_factor_col": season_res.season_factor_col,
        "clean_suffix": season_res.cleaned_suffix,
        "stock_constants": stock_constants,
        "stock_channel_configs": stock_channel_configs,
    }
    return ctx
