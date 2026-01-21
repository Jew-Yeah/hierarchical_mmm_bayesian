# baseline.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from core.preprocessing import load_dataset
from core.seasonality import stage_apply_weekly_seasonality, detect_lead_columns_weekly


def build_context(dataset_name: str, data_root: str = "data", *, smooth_window: int = 3) -> Dict[str, Any]:
    data = load_dataset(dataset_name=dataset_name, data_root=data_root)
    df_weekly = data["combined_weekly"]

    lead_cols = detect_lead_columns_weekly(df_weekly)
    season_res = stage_apply_weekly_seasonality(
        combined_weekly=df_weekly,
        demand_col="demand_queries",
        lead_cols=lead_cols,
        smooth_window=smooth_window,
        clip_range=(0.6, 1.6),
    )

    ctx = {
        "dataset_name": dataset_name,
        "data_root": data_root,
        "raw": data,
        "df_weekly": season_res.weekly,  # уже с season_factor и __leads_clean
        "season_factor_col": season_res.season_factor_col,
        "clean_suffix": season_res.cleaned_suffix,
    }
    return ctx


def main():
    data_dir = Path("data")
    folder_names = [p.name for p in data_dir.iterdir() if p.is_dir()]
    print("Выберите датасет для работы: ")

    for elem in folder_names:
        print(f"- {elem}")

    dataset_name = input("Введите имя датасета из папки data: ")
    ctx = build_context(dataset_name)

    df = ctx["df_weekly"]
    print(df)
    print([c for c in df.columns if c.endswith("__leads_clean")][:10])


if __name__ == "__main__":
    main()
