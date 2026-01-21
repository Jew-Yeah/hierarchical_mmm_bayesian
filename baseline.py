# baseline.py
"""Convenience entrypoints.

This repo is organized as:
- core/: reusable stages (preprocessing, seasonality, fitting, etc.)
- pipelines/: user-facing workflows (forecast_channel, backtest, ...)
- main.py: CLI / interactive runner

baseline.py stays import-friendly (no interactive prompts).
"""

from __future__ import annotations

from typing import Any, Dict, Union
from pathlib import Path

from core.context_builder import build_context


__all__ = ["build_context"]


if __name__ == "__main__":
    # Minimal smoke-run for local debugging
    data_dir = Path("data")
    datasets = [p.name for p in data_dir.iterdir() if p.is_dir()]
    print("Datasets:", datasets)
    if datasets:
        ctx = build_context(datasets[0])
        print(ctx["df_weekly"].head())
