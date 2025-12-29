# comparison.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from fastdtw import fastdtw
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="divide by zero")

# ——— Веса групп признаков ———
GROUP_WEIGHTS = {
    "key_angles": 0.40,       # shoulder_L/R, x_factor — ключевые
    "other_angles": 0.25,     # elbow, hip, knee
    "velocities": 0.15,
    "accelerations": 0.10,
    "entropies": 0.10
}

# Вес внутри ключевых углов: X-фактор важнее на 50%
KEY_ANGLE_WEIGHTS = {
    "shoulder_left_angle": 1.0,
    "shoulder_right_angle": 1.0,
    "x_factor_angle": 1.5
}
KEY_ANGLE_TOTAL = sum(KEY_ANGLE_WEIGHTS.values())

# ——— Веса метрик внутри одного временного ряда ———
METRIC_WEIGHTS = {
    "dtw_norm": 0.35,      # устойчив к временному сдвигу и локальной деформации
    "wasserstein": 0.25,   # распределение значений (непараметрически)
    "spearman": 0.25,      # форма траектории (без линейных предпосылок)
    "rmse_norm": 0.15,     # штраф за систематическую ошибку (после выравнивания по фазе)
}


def _clean_series(a):
    """Очистка ОДНОГО ряда: удаление NaN, inf, константных данных. Возвращает None при недостатке данных."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    a = a[~np.isinf(a)]
    if len(a) < 3:
        return None
    if np.isclose(np.std(a), 0.0, atol=1e-4):
        return None
    return a


def _interpolate_to_same_length(a, b, target_len=60):
    """Линейная интерполяция двух рядов к общей длине (мин. 10 точек)."""
    n = max(len(a), len(b), target_len, 10)
    t_a = np.linspace(0, 1, len(a))
    t_b = np.linspace(0, 1, len(b))
    a_i = np.interp(np.linspace(0, 1, n), t_a, a)
    b_i = np.interp(np.linspace(0, 1, n), t_b, b)
    return a_i, b_i


def _get_scale(col, model_series):
    """Адаптивный масштаб для нормировки: физиологически значимый минимум + σ модели."""
    sigma = np.std(model_series) or 1e-3
    if "_angle" in col:
        return max(sigma, 1.0)      # 1° — минимальная значимая разница
    elif "_velocity" in col:
        return max(sigma, 5.0)      # 5°/с
    elif "_acceleration" in col:
        return max(sigma, 20.0)     # 20°/s²
    else:  # entropy
        return 1.0


def _score_series(model_series, user_series, col):
    """
    Оценка одного временного ряда [0, 1] с учётом временного сдвига и шума.
    """
    # 1. Очистка каждого ряда независимо
    a_clean = _clean_series(model_series)
    b_clean = _clean_series(user_series)
    if a_clean is None or b_clean is None:
        return np.nan

    # 2. Интерполяция к общей длине
    a_i, b_i = _interpolate_to_same_length(a_clean, b_clean)

    # 3. Масштаб
    scale = _get_scale(col, a_clean)

    # 4. Метрики
    # DTW (ограничение окна — 20% для физ. правдоподобия)
    try:
        dist, _ = fastdtw(a_i, b_i, radius=max(1, len(a_i) // 5))
        dtw_norm = max(0.0, min(1.0, 1.0 - dist / (len(a_i) * scale)))
    except Exception:
        dtw_norm = 0.0

    # Wasserstein (Earth Mover’s Distance)
    try:
        w_dist = wasserstein_distance(a_i, b_i)
        wasserstein_score = max(0.0, min(1.0, 1.0 - w_dist / scale))
    except Exception:
        wasserstein_score = 0.0

    # Spearman rank correlation
    try:
        rho, _ = spearmanr(a_i, b_i)
        spearman_score = (1.0 + rho) / 2.0 if not np.isnan(rho) else 0.0
    except Exception:
        spearman_score = 0.0

    # RMSE с учётом фазового сдвига (через кросс-корреляцию)
    try:
        xc = np.correlate(a_i - np.mean(a_i), b_i - np.mean(b_i), mode='full')
        offset = np.argmax(xc) - (len(a_i) - 1)
        b_shifted = np.roll(b_i, offset)
        rmse = np.sqrt(np.mean((a_i - b_shifted) ** 2))
        rmse_norm = max(0.0, min(1.0, 1.0 - rmse / scale))
    except Exception:
        rmse_norm = 0.0

    # Взвешенная комбинация
    return (
        METRIC_WEIGHTS["dtw_norm"] * dtw_norm +
        METRIC_WEIGHTS["wasserstein"] * wasserstein_score +
        METRIC_WEIGHTS["spearman"] * spearman_score +
        METRIC_WEIGHTS["rmse_norm"] * rmse_norm
    )


def compare_model_user(model_df, user_df):
    """
    Сравнение модели и пользователя по всем доступным признакам.
    Возвращает: (общий_%, детали_по_группам)
    """
    # ——— Фильтрация колонок ———
    cols = [
        c for c in model_df.columns
        if c not in ['frame_number', 'avg_visibility']
        and np.issubdtype(model_df[c].dtype, np.number)
        and c in user_df.columns
    ]

    # ——— Агрегация по группам ———
    group_scores = {g: [] for g in GROUP_WEIGHTS}
    group_weights = {g: [] for g in GROUP_WEIGHTS}

    for col in cols:
        m_val = model_df[col].dropna().values.astype(float)
        u_val = user_df[col].dropna().values.astype(float)
        if len(m_val) < 3 or len(u_val) < 3:
            continue

        score = _score_series(m_val, u_val, col)
        if np.isnan(score):
            continue

        # Определение группы и веса
        if col in KEY_ANGLE_WEIGHTS:
            group = "key_angles"
            w = KEY_ANGLE_WEIGHTS[col] / KEY_ANGLE_TOTAL
        elif "_angle" in col:
            group = "other_angles"
            w = 1.0
        elif "_velocity" in col:
            group = "velocities"
            w = 1.0
        elif "_acceleration" in col:
            group = "accelerations"
            w = 1.0
        elif "_entropy" in col:
            group = "entropies"
            w = 1.0
        else:
            continue

        group_scores[group].append(score)
        group_weights[group].append(w)

    # ——— Взвешенное усреднение ———
    group_results = {}
    total_score = 0.0

    for group in GROUP_WEIGHTS:
        scores = np.array(group_scores[group])
        weights = np.array(group_weights[group])

        if len(scores) == 0:
            avg = 0.0
        else:
            avg = np.average(scores, weights=weights) if weights.sum() > 0 else np.mean(scores)

        group_weight = GROUP_WEIGHTS[group]
        group_results[group] = (float(avg), group_weight)
        total_score += avg * group_weight

    return round(total_score * 100, 1), group_results