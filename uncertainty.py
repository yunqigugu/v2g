"""
uncertainty.py — B 组四个不确定性模块
B1 光伏：RF 残差条件回归 + 24 场景 + 25% 分位
B2 负荷：同构，取 75% 分位
B3 电价：RF 预测 + 高斯噪声
B4 EV：三峰到达 / 截断停留 / RF SOC / Beta 意愿
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from core import synth_load, synth_pv, gen_ev_fleet

_cache = {}


# ============================================================
# 通用工具
# ============================================================
def _features(n, date, forecast):
    t = np.arange(n)
    return np.column_stack([
        t, t // 2,
        np.sin(2 * np.pi * t / n),
        np.cos(2 * np.pi * t / n),
        np.full(n, date.dayofweek),
        np.full(n, date.month),
        forecast,
    ])


def _synth_pv_residuals(cfg, n_hist=30):
    """合成历史光伏残差。"""
    rng = np.random.default_rng(cfg["seed"])
    residuals, forecasts, dates = [], [], []
    for d in range(n_hist):
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)
        fc = synth_pv(d, cfg)
        r = rng.normal(0, 80, 48)
        h = np.arange(48) / 2.0
        r[(h >= 6) & (h <= 18)] *= 3.0
        residuals.append(r)
        forecasts.append(fc)
        dates.append(date)
    return {"residuals": np.array(residuals),
            "forecasts": np.array(forecasts),
            "dates": dates}


def _synth_load_residuals(cfg, n_hist=30):
    rng = np.random.default_rng(cfg["seed"] + 1)
    residuals, forecasts, dates = [], [], []
    for d in range(n_hist):
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)
        fc = synth_load(d, cfg)
        r = rng.normal(0, 120, 48)
        residuals.append(r)
        forecasts.append(fc)
        dates.append(date)
    return {"residuals": np.array(residuals),
            "forecasts": np.array(forecasts),
            "dates": dates}


def _train_rf(hist, cfg):
    X_rows, y_rows = [], []
    for i, d in enumerate(hist["dates"]):
        X_rows.append(_features(48, d, hist["forecasts"][i]))
        y_rows.append(hist["residuals"][i])
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    rf = RandomForestRegressor(n_estimators=30, max_depth=5,
                               random_state=cfg["seed"], n_jobs=-1)
    rf.fit(X, y)
    return rf


def _sample_scenarios(rf, forecast, date, n_scen, cfg):
    """从 RF 森林里采样残差，生成场景。"""
    X = _features(len(forecast), date, forecast)
    preds = np.array([t.predict(X) for t in rf.estimators_])
    rng = np.random.default_rng(cfg["seed"] + date.day)
    if preds.shape[0] > n_scen:
        idx = rng.choice(preds.shape[0], size=n_scen, replace=False)
    else:
        idx = np.arange(preds.shape[0])
    return np.array([np.clip(forecast + preds[i], 0, None) for i in idx])


def _get_rf(kind, cfg):
    key = f"rf_{kind}"
    if key in _cache:
        return _cache[key]
    hist = _synth_pv_residuals(cfg) if kind == "pv" else _synth_load_residuals(cfg)
    _cache[key] = _train_rf(hist, cfg)
    return _cache[key]


# ============================================================
# B1 光伏不确定性
# ============================================================
def b1_pv_uncertainty(pv_forecast, day_idx, cfg):
    """取 25% 分位作为保守预测。"""
    rf = _get_rf("pv", cfg)
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_idx)
    scen = _sample_scenarios(rf, pv_forecast, date,
                             cfg["B_pv_n_scen"], cfg)
    return np.quantile(scen, cfg["B_pv_quantile"], axis=0)


# ============================================================
# B2 负荷不确定性
# ============================================================
def b2_load_uncertainty(load_forecast, day_idx, cfg):
    """取 75% 分位作为保守预测。"""
    rf = _get_rf("load", cfg)
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_idx)
    scen = _sample_scenarios(rf, load_forecast, date,
                             cfg["B_load_n_scen"], cfg)
    return np.quantile(scen, cfg["B_load_quantile"], axis=0)


# ============================================================
# B3 电价不确定性
# ============================================================
def b3_price_uncertainty(price, day_idx, cfg):
    """RF 预测（此处用模板做基线）+ 高斯噪声。"""
    rng = np.random.default_rng(cfg["seed"] + day_idx * 53 + 3)
    noise = rng.normal(0, cfg["B_price_noise"], len(price))
    return np.clip(price + noise, 0.01, None)


# ============================================================
# B4 EV 行为不确定性
# ============================================================
def b4_ev_uncertainty(fleet, day_idx, cfg):
    """重采样 EV 车队（gen_ev_fleet 内部已含三峰/截断/Beta）。"""
    return gen_ev_fleet(day_idx + 1000, cfg)


# ============================================================
# 总入口：按顺序叠加 B1→B2→B3→B4
# ============================================================
def apply_b_uncertainty(price, load, pv, fleet, day_idx, cfg):
    if cfg.get("B1_pv", False):
        pv = b1_pv_uncertainty(pv, day_idx, cfg)
    if cfg.get("B2_load", False):
        load = b2_load_uncertainty(load, day_idx, cfg)
    if cfg.get("B3_price", False):
        price = b3_price_uncertainty(price, day_idx, cfg)
    if cfg.get("B4_ev", False):
        fleet = b4_ev_uncertainty(fleet, day_idx, cfg)
    return price, load, pv, fleet
