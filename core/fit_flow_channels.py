from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd


def _geometric_adstock(spend: np.ndarray, lam: float) -> np.ndarray:
    """E_t = lam * E_{t-1} + spend_t"""
    spend = np.asarray(spend, dtype=float)
    E = np.zeros_like(spend, dtype=float)
    prev = 0.0
    for i, s in enumerate(spend):
        prev = lam * prev + (0.0 if not np.isfinite(s) else float(s))
        E[i] = prev
    return E


def _steady_state_effective(spend_weekly: np.ndarray, lam: float) -> np.ndarray:
    """
    Для статической кривой y(x) при постоянном weekly spend = x:
      E_ss = x / (1 - lam)
    (если lam=0 -> E=x).
    """
    spend_weekly = np.asarray(spend_weekly, dtype=float)
    denom = max(1e-6, 1.0 - float(lam))
    return np.clip(spend_weekly, 0.0, np.inf) / denom


def _make_b_grid(E: np.ndarray, n: int = 45) -> np.ndarray:
    """
    Сетка b вокруг 1/x_knee, где x_knee = median(E>0).
    """
    E = np.asarray(E, dtype=float)
    Epos = E[np.isfinite(E) & (E > 0)]
    x_knee = float(np.median(Epos)) if len(Epos) else 1.0
    x_knee = max(1e-6, x_knee)
    b_min = 0.05 / x_knee
    b_max = 20.0 / x_knee
    return np.exp(np.linspace(np.log(b_min), np.log(b_max), n))


def _fit_ab_closed_form(E: np.ndarray, y: np.ndarray, b_grid: np.ndarray) -> Tuple[float, float, float]:
    """
    y ≈ a * log(1 + bE)
    Для каждого b: a = (x^T y)/(x^T x), x = log(1 + bE)
    Выбираем b по минимуму MSE.
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(E) & np.isfinite(y)
    E = E[m]
    y = y[m]
    if len(E) < 4:
        return 0.0, 1.0, float("inf")

    best_mse = float("inf")
    best_a = 0.0
    best_b = 1.0

    for b in b_grid:
        x = np.log1p(np.maximum(0.0, b * E))
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            continue
        a = float(np.dot(x, y) / denom)
        a = max(0.0, a)
        yhat = a * x
        mse = float(np.mean((y - yhat) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_a = a
            best_b = float(b)

    return float(best_a), float(best_b), float(best_mse)


def _choose_holdout(n: int) -> int:
    return max(3, min(8, n // 5))


def fit_flow_channel(
    df: pd.DataFrame,
    channel: str,
    budget_col: str,
    leads_col: str,
    save_dir: str | None = None,
    show_plots: bool = False,
    lambda_grid: List[float] | None = None,
) -> Dict[str, Any]:
    """
    FLOW:
      1) Подбор lambda ∈ [0,1) step=0.1 + (a_mean,b_mean) с holdout.
      2) Фиксируем lambda_best.
      3) Делим точки по сравнению с mean-кривой на истории:
           below: y <= yhat_mean_hist
           above: y >= yhat_mean_hist
      4) На below и above ЗАНОВО подбираем (a,b) (оба!), получая lower/upper.

    Возвращает predictors для графика x->y, где x=weekly spend:
      predict_mean(x), predict_lower(x), predict_upper(x)
    """
    if lambda_grid is None:
        lambda_grid = [round(x, 1) for x in np.arange(0.0, 1.0, 0.1).tolist()]

    spend = df[budget_col].astype(float).fillna(0.0).to_numpy()
    y = df[leads_col].astype(float).fillna(0.0).to_numpy()
    n = len(spend)

    h = _choose_holdout(n)
    idx_train = np.arange(0, max(0, n - h))
    idx_test = np.arange(max(0, n - h), n)

    best = {"lam": 0.0, "a": 0.0, "b": 1.0, "mse": float("inf")}

    # 1) Выбор lambda + mean(a,b)
    for lam in lambda_grid:
        lam = float(lam)
        E_hist = _geometric_adstock(spend, lam)
        b_grid = _make_b_grid(E_hist[idx_train], n=45)
        a, b, _ = _fit_ab_closed_form(E_hist[idx_train], y[idx_train], b_grid)

        # holdout mse
        if len(idx_test):
            x_test = np.log1p(np.maximum(0.0, b * E_hist[idx_test]))
            yhat_test = a * x_test
            mse = float(np.mean((y[idx_test] - yhat_test) ** 2))
        else:
            yhat_all = a * np.log1p(np.maximum(0.0, b * E_hist))
            mse = float(np.mean((y - yhat_all) ** 2))

        if mse < best["mse"]:
            best.update({"lam": lam, "a": float(a), "b": float(b), "mse": mse})

    lam_best = float(best["lam"])
    a_mean = float(best["a"])
    b_mean = float(best["b"])

    # 2) mean на истории (для разбиения точек)
    E_hist_best = _geometric_adstock(spend, lam_best)
    yhat_hist_mean = a_mean * np.log1p(np.maximum(0.0, b_mean * E_hist_best))

    below = y <= (yhat_hist_mean + 1e-12)
    above = y >= (yhat_hist_mean - 1e-12)

    # 3) Перефит (a,b) на подвыборках (оба параметра!)
    def _fit_subset(mask: np.ndarray) -> Tuple[float, float]:
        if int(np.sum(mask)) < 6:
            # мало точек — fallback к mean
            return a_mean, b_mean
        E_sub = E_hist_best[mask]
        y_sub = y[mask]
        b_grid = _make_b_grid(E_sub, n=40)
        a, b, _ = _fit_ab_closed_form(E_sub, y_sub, b_grid)
        return float(a), float(b)

    a_low, b_low = _fit_subset(below)
    a_high, b_high = _fit_subset(above)

    # predictors по weekly spend -> steady-state E -> R(E)
    def _predict(spend_weekly: np.ndarray, a: float, b: float) -> np.ndarray:
        E = _steady_state_effective(spend_weekly, lam_best)
        return a * np.log1p(np.maximum(0.0, b * E))

    def predict_mean(x: np.ndarray) -> np.ndarray:
        return _predict(x, a_mean, b_mean)

    def predict_lower(x: np.ndarray) -> np.ndarray:
        return _predict(x, a_low, b_low)

    def predict_upper(x: np.ndarray) -> np.ndarray:
        return _predict(x, a_high, b_high)

    return {
        "channel": channel,
        "budget_col": budget_col,
        "leads_col": leads_col,
        "lambda_grid": lambda_grid,
        "lambda_best": lam_best,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "a_low": a_low,
        "b_low": b_low,
        "a_high": a_high,
        "b_high": b_high,
        "mse_holdout": float(best["mse"]),
        "predict_mean": predict_mean,
        "predict_lower": predict_lower,
        "predict_upper": predict_upper,
    }
