from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.context_builder import build_context
from core.fit_flow_channels import fit_flow_channel
from core.fit_stock_channels import fit_stock_channel_with_cpt


def _period_label_weekly(df) -> str:
    if "week_start" not in df.columns or df["week_start"].isna().all():
        return "по неделям"
    d0 = df["week_start"].min()
    d1 = df["week_start"].max()
    return f"по неделям ({d0:%Y-%m-%d} … {d1:%Y-%m-%d})"


def _detect_channels(df_columns: List[str], prefix: str, suffix: str) -> List[str]:
    out = []
    seen = set()
    for c in df_columns:
        if c.startswith(prefix) and c.endswith(suffix):
            ch = c[len(prefix) : -len(suffix)]
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
    return out


def forecast_channel(
    dataset_name: str,
    *,
    data_root: str | Path = "data",
    save_dir: str | Path = "save_images",
    show_plots: bool = True,
) -> Dict[str, Any]:
    """
    Пайплайн:
      dataset_name -> build_context -> fit_flow + fit_stock -> графики.

    Графики подписаны как "по неделям" + диапазон дат week_start.
    """
    save_dir = Path(save_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(dataset_name=dataset_name, data_root=str(data_root))
    df = ctx["df_weekly"]

    period = _period_label_weekly(df)

    # Авто-детект каналов из колонок spend
    flow_channels = _detect_channels(df.columns.tolist(), prefix="flow__", suffix="__spend")
    stock_channels = _detect_channels(df.columns.tolist(), prefix="stock__", suffix="__spend")

    results: Dict[str, Any] = {"flow": {}, "stock": {}, "meta": {"dataset": dataset_name}}

    # ---------------- FLOW ----------------
    for ch in flow_channels:
        spend_col = f"flow__{ch}__spend"
        leads_base = f"flow__{ch}__leads"
        leads_clean = f"flow__{ch}__leads_clean"
        y_col = leads_clean if leads_clean in df.columns else leads_base

        if spend_col not in df.columns or y_col not in df.columns:
            print(f"[FLOW:{ch}] пропуск: нет {spend_col} или {y_col}")
            continue

        res = fit_flow_channel(
            df=df,
            channel=ch,
            budget_col=spend_col,
            leads_col=y_col,
        )
        results["flow"][ch] = res

        # Печать функций в консоль
        print(f"\nFLOW {ch}:")
        print(f"  lambda_best={res['lambda_best']:.3f}")
        print(f"  mean : a={res['a_mean']:.6g}, b={res['b_mean']:.6g}")
        print(f"  lower: a={res['a_low']:.6g},  b={res['b_low']:.6g}   (консервативная нижняя)")
        print(f"  upper: a={res['a_high']:.6g}, b={res['b_high']:.6g}")

        # График: x = spend (в неделю), y = leads/week
        x = df[spend_col].astype(float).fillna(0.0).to_numpy()
        y = df[y_col].astype(float).fillna(0.0).to_numpy()

        x_max = max(float(np.nanmax(x)) if len(x) else 1.0, 1600)
        x_grid = np.linspace(0.0, max(1.0, x_max), 160)

        y_mean = res["predict_mean"](x_grid)
        y_low = res["predict_lower"](x_grid)
        y_up = res["predict_upper"](x_grid)

        # сортируем точки для отображения "последовательно по x"
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_s, y_s, label=f"Факт ({period})", alpha=0.85)

        # Разные цвета (попросил явно)
        plt.plot(x_grid, y_mean, linewidth=2.2, color="tab:blue", label="Средний прогноз")
        plt.plot(x_grid, y_low, linewidth=2.0, color="tab:orange", linestyle="--",
                 label="Консервативная нижняя оценка")
        plt.plot(x_grid, y_up, linewidth=2.0, color="tab:green", linestyle=":",
                 label="Верхняя оценка (эвристика)")

        plt.xlabel("Расходы в неделю, ₽")
        plt.ylabel("Лиды за неделю (очищенные от сезонности)" if y_col.endswith("__leads_clean") else "Лиды за неделю")
        plt.title(f"FLOW канал '{ch}': лиды от расходов\n{period}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"flow_{ch}_weekly.png", dpi=170)

    # ---------------- STOCK ----------------
    stock_cfgs = ctx.get("stock_channel_configs", {}) or {}

    for ch in stock_channels:
        spend_col = f"stock__{ch}__spend"
        leads_base = f"stock__{ch}__leads"
        leads_clean = f"stock__{ch}__leads_clean"
        y_col = leads_clean if leads_clean in df.columns else leads_base

        if spend_col not in df.columns or y_col not in df.columns:
            print(f"[STOCK:{ch}] пропуск: нет {spend_col} или {y_col}")
            continue

        active_units_col = f"stock__{ch}__active_units"

        res = fit_stock_channel_with_cpt(
            df=df,
            channel=ch,
            active_units_col=active_units_col,
            leads_col=y_col,
            stock_channel_configs=stock_cfgs,
            spend_col_fallback=spend_col,
        )
        results["stock"][ch] = res

        # Печать распределения в консоль
        print(f"\nSTOCK {ch}:")
        print(f"  dist={res['dist_name']}, mu={res['mu']:.6g}, sigma={res['sigma']:.6g}")
        print(f"  mean curve: a={res['a_mean']:.6g}, b={res['b_mean']:.6g}")

        units_col_used = res["active_units_col"]
        u = df[units_col_used].astype(float).fillna(0.0).to_numpy()
        y = df[y_col].astype(float).fillna(0.0).to_numpy()

        u_max = max(float(np.nanmax(u)) if len(u) else 1.0,300)
        u_grid = np.linspace(0.0, max(1.0, u_max), 300)

        m = res["predict_mean"](u_grid)

        # уровни вероятности
        levels = [
            (0.6827, "68.27% (±1σ)", "tab:green"),
            (0.9545, "95.45% (±2σ)", "tab:orange"),
            (0.9973, "99.73% (±3σ)", "tab:red"),
        ]

        idx = np.argsort(u)
        u_s = u[idx]
        y_s = y[idx]

        plt.figure(figsize=(10, 6))
        plt.scatter(u_s, y_s, label=f"Факт ({period})", alpha=0.85)

        # mean — отдельный цвет
        plt.plot(u_grid, m, linewidth=2.2, color="tab:blue", label="Средний прогноз")

        # интервальные ленты: одним цветом для верх/низ одного уровня
        for level, label, color in levels:
            low, high = res["predict_interval"](u_grid, level)
            plt.plot(u_grid, low, color=color, linewidth=1.6, linestyle="--")
            plt.plot(u_grid, high, color=color, linewidth=1.6, linestyle="--")
            plt.fill_between(u_grid, low, high, color=color, alpha=0.10, label=label)

        plt.xlabel("Активные рекламные юниты (машины)")
        plt.ylabel("Лиды за неделю (очищенные от сезонности)" if y_col.endswith("__leads_clean") else "Лиды за неделю")
        plt.title(f"STOCK канал '{ch}': лиды от числа активных юнитов\n{period}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"stock_{ch}_weekly.png", dpi=170)

    if show_plots:
        plt.show()

    return results
