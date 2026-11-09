from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_NAMES = ['trend', 'season_sin', 'season_cos', 'lag_1', 'lag_4', 'lag_13', 'lag_52']
RIDGE_ALPHA = 8.0


def feature_row(t: int, history: list[float], annual_period: float = 52.18) -> np.ndarray:
    def lag(k: int) -> float:
        return float(history[-k]) if len(history) >= k else float(np.mean(history))
    return np.array([t / 52.0, np.sin(2*np.pi*t/annual_period), np.cos(2*np.pi*t/annual_period),
                     lag(1), lag(4), lag(13), lag(52)], dtype=float)


def fit_series(values: np.ndarray, ridge_alpha: float = RIDGE_ALPHA) -> dict:
    """Fit a per-SKU ridge regression using time, annual seasonality, and demand lags."""
    values = np.asarray(values, dtype=float)
    if len(values) < 60:
        raise ValueError('At least 60 weekly observations are required per SKU.')
    X = np.vstack([feature_row(t, values[:t].tolist()) for t in range(52, len(values))])
    y = values[52:]
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(y.mean())
    y_scale = float(y.std()) or 1.0
    Z = (X - x_mean) / x_scale
    target = (y - y_mean) / y_scale
    design = np.column_stack([np.ones(len(Z)), Z])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    residuals = target - design @ beta
    residual_std = float(np.std(residuals) * y_scale)
    return {'x_mean': x_mean, 'x_scale': x_scale, 'y_mean': y_mean,
            'y_scale': y_scale, 'beta': beta, 'residual_std': residual_std,
            'n_train': int(len(y))}


def predict_next(params: dict, t: int, history: list[float]) -> float:
    x = feature_row(t, history)
    z = (x - params['x_mean']) / params['x_scale']
    y_scaled = params['beta'][0] + z @ params['beta'][1:]
    return max(0.0, float(y_scaled * params['y_scale'] + params['y_mean']))


def forecast_recursive(params: dict, history: list[float], horizon: int) -> list[float]:
    values = list(map(float, history))
    output = []
    for _ in range(horizon):
        pred = predict_next(params, len(values), values)
        output.append(pred)
        values.append(pred)
    return output


def evaluate_holdout(values: np.ndarray, holdout: int = 12) -> dict:
    train, actual = values[:-holdout], values[-holdout:]
    params = fit_series(train)
    pred = np.array(forecast_recursive(params, train.tolist(), holdout))
    mae = float(np.mean(np.abs(actual - pred)))
    wape = float(np.sum(np.abs(actual - pred)) / max(1.0, float(np.sum(actual))))
    naive = np.full(holdout, float(np.mean(train[-4:])))
    naive_mae = float(np.mean(np.abs(actual - naive)))
    naive_wape = float(np.sum(np.abs(actual - naive)) / max(1.0, float(np.sum(actual))))
    return {'mae': mae, 'wape': wape, 'naive_mae': naive_mae,
            'naive_wape': naive_wape, 'predictions': pred.tolist(), 'actual': actual.tolist()}


def save_model(params_by_sku: dict, path: Path, metadata: dict) -> None:
    serializable = {}
    for sku, p in params_by_sku.items():
        serializable[sku] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'metadata': metadata, 'models': serializable}, indent=2), encoding='utf-8')


def load_model(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    for p in obj['models'].values():
        for k in ('x_mean','x_scale','beta'):
            p[k] = np.array(p[k], dtype=float)
    return obj

