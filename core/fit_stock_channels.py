from __future__ import annotations

from typing import Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd


# ---------- Helpers: active units reconstruction (если колонки нет) ----------

def _ensure_active_units(
    df: pd.DataFrame,
    channel: str,
    active_units_col: str,
    spend_col: str,
    stock_channel_configs: Dict[str, Any] | None,
) -> str:
    """
    Если active_units_col отсутствует — строим активные юниты из spend + configs:
      active[t] = active[t-1]*decay + spend[t]/cost_per_unit
    """
    if active_units_col in df.columns:
        return active_units_col

    if spend_col not in df.columns:
        raise KeyError(f"[stock:{channel}] нет spend_col={spend_col} и нет {active_units_col}")

    cfg = (stock_channel_configs or {}).get(channel, {}) if isinstance(stock_channel_configs, dict) else {}

    cost_per_unit = float(cfg.get("cost_per_unit", cfg.get("unit_cost", np.nan)))
    if not np.isfinite(cost_per_unit) or cost_per_unit <= 0:
        raise ValueError(
            f"[stock:{channel}] Не задана стоимость юнита cost_per_unit в stock_channel_configs[{channel}]."
        )

    decay = float(cfg.get("decay", 0.995))
    if not (0.0 < decay <= 1.0):
        decay = 0.995

    initial_units = float(cfg.get("initial_units", cfg.get("units_start", 0.0)))
    if initial_units < 0:
        initial_units = 0.0

    spend = df[spend_col].astype(float).fillna(0.0).to_numpy()
    purchases = spend / cost_per_unit

    active = np.zeros_like(purchases, dtype=float)
    prev = initial_units
    for i, u in enumerate(purchases):
        prev = prev * decay + float(u)
        active[i] = prev

    df[active_units_col] = active
    return active_units_col


# ---------- Mean curve fit: y ≈ a ln(1 + b*S) ----------

def _make_b_grid(S: np.ndarray, n: int = 45) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    Spos = S[np.isfinite(S) & (S > 0)]
    knee = float(np.median(Spos)) if len(Spos) else 1.0
    knee = max(1e-6, knee)
    b_min = 0.05 / knee
    b_max = 20.0 / knee
    return np.exp(np.linspace(np.log(b_min), np.log(b_max), n))


def _fit_ab_curve(S: np.ndarray, y: np.ndarray, b_grid: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit mean curve: y ≈ a * log(1 + b*S)
    For each b, optimal a in L2: a = (x^T y)/(x^T x), x = log(1 + b*S)
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(S) & np.isfinite(y)
    S = S[m]
    y = y[m]
    if len(S) < 4:
        return 0.0, 1.0, float("inf")

    best_mse = float("inf")
    best_a = 0.0
    best_b = 1.0

    for b in b_grid:
        x = np.log1p(np.maximum(0.0, b * S))
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


# ---------- CLT variance calibration (per-unit touch variance) ----------

def _calibrate_sigma_unit_clt(
    S: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
) -> float:
    """
    Мы моделируем касания X|S ~ Normal(S*mu, S*sigma^2), но фиксируем mu=1 (масштаб).
    Тогда mean leads: m(S) = a ln(1 + b * (S*mu)) = a ln(1 + bS)

    Для дисперсии лидов используем дельта-метод:
      Var(Y|S) ≈ (f'(E[X]))^2 Var(X)
      f(x)=a ln(1+ b x), f'(x)= a*b/(1+b x)
      E[X]=S (так как mu=1), Var(X)=S*sigma_unit^2

    => Var(Y|S) ≈ [a*b/(1+bS)]^2 * S * sigma_unit^2

    Оцениваем sigma_unit^2 робастно по остаткам:
      r^2 ≈ V(S) * sigma_unit^2, где V(S)=[a*b/(1+bS)]^2 * S
      sigma_unit^2 ≈ median(r^2 / V(S))
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(S) & np.isfinite(y) & (S >= 0)
    S = S[m]
    y = y[m]
    if len(S) < 6:
        return 1.0  # fallback

    mean_y = a * np.log1p(np.maximum(0.0, b * S))
    r = y - mean_y

    # V(S) factor
    denom = (1.0 + b * S)
    deriv = (a * b) / np.maximum(1e-12, denom)
    V = (deriv ** 2) * np.maximum(0.0, S)

    good = V > 1e-12
    if np.sum(good) < 4:
        return 1.0

    ratio = (r[good] ** 2) / V[good]
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return 1.0

    sigma2 = float(np.median(ratio))
    sigma2 = max(1e-12, sigma2)
    return float(np.sqrt(sigma2))


# ---------- Main API ----------

def fit_stock_channel_with_cpt(
    df: pd.DataFrame,
    channel: str,
    active_units_col: str,
    leads_col: str,
    save_dir: str | None = None,
    show_plots: bool = False,
    stock_channel_configs: Dict[str, Any] | None = None,
    spend_col_fallback: str | None = None,
) -> Dict[str, Any]:
    """
    STOCK (CLT-версия):

    Модель:
      X|S ~ Normal(S*mu, S*sigma^2), mu фиксируем = 1 (нормировка шкалы касаний)
      Y = a * ln(1 + b*X)

    В MVP используем:
      mean(S) = a ln(1 + bS)
      sigma_unit калибруем через дельта-метод и остатки.
      Доверительные интервалы 1/2/3σ строим как:
        mean(S) ± k * std_Y(S)
      где std_Y(S) = |f'(E[X])| * sqrt(S) * sigma_unit.

    Возвращаем:
      dist_name: "clt_normal_touches"
      mu: 1.0  (среднее касаний на 1 юнит в нормированной шкале)
      sigma: sigma_unit (std касаний на 1 юнит в той же шкале)
      predict_mean(S)
      predict_interval(S, level) -> (low, high), level ∈ {0.6827,0.9545,0.9973}
    """
    # ensure active units
    if active_units_col not in df.columns:
        if spend_col_fallback is None:
            spend_col_fallback = f"stock__{channel}__spend"
        active_units_col = _ensure_active_units(
            df=df,
            channel=channel,
            active_units_col=active_units_col,
            spend_col=spend_col_fallback,
            stock_channel_configs=stock_channel_configs,
        )

    S = df[active_units_col].astype(float).fillna(0.0).to_numpy()
    y = df[leads_col].astype(float).fillna(0.0).to_numpy()

    # fit mean curve in S
    b_grid = _make_b_grid(S, n=45)
    a_mean, b_mean, mse = _fit_ab_curve(S, y, b_grid)

    def predict_mean(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return a_mean * np.log1p(np.maximum(0.0, b_mean * np.maximum(0.0, s)))

    # CLT per-unit sigma (touches)
    sigma_unit = _calibrate_sigma_unit_clt(S, y, a_mean, b_mean)

    # std of Y via delta method
    def _std_y(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        s_pos = np.maximum(0.0, s)
        # E[X]=S (mu=1)
        denom = 1.0 + b_mean * s_pos
        deriv = (a_mean * b_mean) / np.maximum(1e-12, denom)  # f'(E[X])
        std_x = np.sqrt(s_pos) * sigma_unit
        return np.abs(deriv) * std_x

    def _k_from_level(level: float) -> float:
        if abs(level - 0.6827) < 1e-3:
            return 1.0
        if abs(level - 0.9545) < 1e-3:
            return 2.0
        if abs(level - 0.9973) < 1e-3:
            return 3.0
        return 2.0

    def predict_interval(s: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        m = predict_mean(s)
        sd = _std_y(s)
        k = _k_from_level(float(level))
        low = np.maximum(0.0, m - k * sd)
        high = np.maximum(0.0, m + k * sd)
        return low, high

    return {
        "channel": channel,
        "active_units_col": active_units_col,
        "leads_col": leads_col,
        # mean curve params
        "a_mean": float(a_mean),
        "b_mean": float(b_mean),
        "mse": float(mse),
        # distribution of touches per unit (normalized scale)
        "dist_name": "clt_normal_touches",
        "mu": 1.0,
        "sigma": float(sigma_unit),
        # predictors
        "predict_mean": predict_mean,
        "predict_interval": predict_interval,
        # for debugging / explanation
        "note": "X|S ~ Normal(S*mu, S*sigma^2), mu fixed to 1 (scale), Y=a*ln(1+bX), intervals via delta method",
    }
