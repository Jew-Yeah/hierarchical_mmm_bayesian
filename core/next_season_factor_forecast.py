import pandas as pd
import numpy as np

def weighted_average_by_month(season_factor_weekly: pd.Series) -> pd.Series:
    """
    Прогнозирует сезонность по месяцу, используя взвешенное среднее сезонных коэффициентов.
    Веса определяются количеством недель в месяце (например, последняя неделя важнее).
    """
    # Рассчитываем количество недель в месяце
    season_factor_monthly = season_factor_weekly.resample('M').mean()

    # Весовая агрегация
    weighted_seasonality = []
    for month, group in season_factor_weekly.resample('M').groups.items():
        # Веса: например, последняя неделя важнее, чем первая
        weights = np.arange(1, len(group)+1)  # веса: от 1 до кол-ва недель в месяце
        weighted_seasonality.append(np.average(group, weights=weights))

    weighted_seasonality_series = pd.Series(weighted_seasonality, index=season_factor_monthly.index)
    return weighted_seasonality_series


def exp_smoothing_for_seasonality(season_factor_weekly: pd.Series, span: int = 4) -> pd.Series:
    """
    Применяет экспоненциальное сглаживание для предсказания сезонности.
    """
    smoothed_seasonality = season_factor_weekly.ewm(span=span, adjust=False).mean()
    return smoothed_seasonality


def forecast_season_factor(
    season_hist: pd.Series,   # Исторические данные сезонности или Wordstat
    horizon_weeks: int = 4,   # Прогноз на сколько недель вперед
    method: str = "moving_average",  # Метод прогноза: "moving_average" или "exp_smoothing"
    window_size: int = 4  # Размер окна для скользящего среднего
) -> pd.Series:
    """
    Прогнозирует сезонный коэффициент на будущее (недели или месяцы)
    на основе исторических данных сезонности.
    """
    if method == "moving_average":
        season_pred = season_hist.rolling(window=window_size, min_periods=1).mean()
    elif method == "exp_smoothing":
        season_pred = season_hist.ewm(span=window_size, adjust=False).mean()
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # Прогноз на будущее (сдвигаем на horizon_weeks вперед)
    forecasted_season_factor = season_pred[-1] * pd.Series([1] * horizon_weeks)  # Прогнозируем одинаково по времени
    forecasted_season_factor.index = pd.date_range(season_hist.index[-1], periods=horizon_weeks + 1, freq="W")[1:]
    
    return forecasted_season_factor


def get_seasonality_from_demand(demand_df: pd.DataFrame, date_col: str = "week_start", season_factor_col: str = "season_factor") -> pd.Series:
    """
    Функция для расчета сезонности на основе данных по запросам demand.
    Это может быть полезно, если данные по запросам используются для определения сезонности.
    """
    # Группируем данные по неделям/месяцам, чтобы получить seasonal factor
    demand_df['date'] = pd.to_datetime(demand_df[date_col])
    demand_df.set_index('date', inplace=True)

    # Рассчитываем сезонный коэффициент как отношение demand/нормализованного demand
    demand_mean = demand_df[season_factor_col].mean()
    demand_df[season_factor_col] = demand_df[season_factor_col] / demand_mean

    return demand_df[season_factor_col]
