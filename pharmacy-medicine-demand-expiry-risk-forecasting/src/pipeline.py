from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .forecast_model import evaluate_holdout, fit_series, forecast_recursive, save_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'evidence'
MODEL = ROOT / 'models' / 'demand_model.json'
HORIZON = 12


def simulate_expiry(lots: pd.DataFrame, forecasts: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Project FEFO consumption and count units left when their lot expires."""
    snapshot = pd.Timestamp(snapshot_date)
    rows = []
    for sku, group in lots.groupby('sku_id'):
        remaining = [{'lot_id': r.lot_id, 'qty': int(r.on_hand_units),
                      'expiry_week': max(1, int(np.ceil((pd.Timestamp(r.expiry_date)-snapshot).days/7)))}
                     for r in group.itertuples()]
        start_units = sum(x['qty'] for x in remaining)
        expired_units = 0
        sku_fc = forecasts[forecasts.sku_id == sku].sort_values('week_start')
        for step, f in enumerate(sku_fc.itertuples(), start=1):
            need = int(np.ceil(float(f.forecast_units)))
            # First-expiry-first-out (FEFO): consume the closest-dated usable lot first.
            for lot in sorted(remaining, key=lambda x: x['expiry_week']):
                if lot['qty'] <= 0 or lot['expiry_week'] < step:
                    continue
                used = min(lot['qty'], need)
                lot['qty'] -= int(np.ceil(used))
                need = max(0.0, need - used)
                if need <= 0:
                    break
            for lot in remaining:
                if lot['expiry_week'] == step and lot['qty'] > 0:
                    expired_units += lot['qty']
                    lot['qty'] = 0
        rows.append({'sku_id': sku, 'starting_on_hand_units': start_units,
                     'forecast_demand_12w': round(float(sku_fc.forecast_units.sum()), 1),
                     'units_expiring_unconsumed_12w': int(expired_units),
                     'expiry_risk_pct_of_starting_stock': round(100*expired_units/max(1,start_units), 1)})
    return pd.DataFrame(rows)


def run_pipeline() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(DATA / 'weekly_demand.csv', parse_dates=['week_start'])
    products = pd.read_csv(DATA / 'products.csv')
    lots = pd.read_csv(DATA / 'inventory_lots.csv', parse_dates=['snapshot_date','expiry_date'])
    snapshot = lots.snapshot_date.min().date().isoformat()
    forecast_dates = pd.date_range(snapshot, periods=HORIZON, freq='W-MON')
    forecasts, params_by_sku, metric_rows = [], {}, []
    for sku, group in sales.groupby('sku_id'):
        group = group.sort_values('week_start')
        values = group.demand_units.to_numpy(dtype=float)
        ev = evaluate_holdout(values, holdout=12)
        metric_rows.append({'sku_id': sku, **{k: ev[k] for k in ('mae','wape','naive_mae','naive_wape')}})
        params = fit_series(values)
        params_by_sku[sku] = params
        future = forecast_recursive(params, values.tolist(), HORIZON)
        for dt, point in zip(forecast_dates, future):
            spread = 1.28 * params['residual_std']
            forecasts.append({'week_start': dt.date().isoformat(), 'sku_id': sku,
                              'forecast_units': round(point, 1),
                              'approx_p10_units': round(max(0.0, point-spread), 1),
                              'approx_p90_units': round(point+spread, 1)})
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(OUT / '12_week_demand_forecast.csv', index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / 'holdout_metrics_by_sku.csv', index=False)
    # Weight SKU WAPE values by holdout demand volume for a cohort-level score.\r\n    # Weighted by actual holdout volume, not full-period volume.
    weights = np.array([sales[(sales.sku_id == sku)].sort_values('week_start').demand_units.tail(12).sum() for sku in metrics.sku_id], dtype=float)
    aggregate_wape = float(np.average(metrics.wape, weights=weights))
    aggregate_naive = float(np.average(metrics.naive_wape, weights=weights))
    summary = {'n_skus': int(sales.sku_id.nunique()), 'history_weeks_per_sku': int(sales.week_start.nunique()),
               'backtest_horizon_weeks': 12, 'forecast_horizon_weeks': HORIZON,
               'aggregate_wape': aggregate_wape, 'naive_4_week_average_wape': aggregate_naive,
               'wape_change_vs_naive_pct': 100*(aggregate_naive-aggregate_wape)/max(aggregate_naive,1e-9),
               'mean_sku_mae': float(metrics.mae.mean()), 'snapshot_date': snapshot}
    (OUT / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    inventory = simulate_expiry(lots, forecast_df, snapshot)
    # Pair inventory exposure with a simple lead-time replenishment estimate.
    inventory['risk_value_estimate'] = 0.0
    inventory['recommended_order_units'] = 0
    product_map = products.set_index('sku_id')
    for i, row in inventory.iterrows():
        p = product_map.loc[row.sku_id]
        sku_fc = forecast_df[forecast_df.sku_id == row.sku_id].sort_values('week_start')
        lead = int(p.lead_time_weeks)
        lead_demand = float(sku_fc.head(lead).forecast_units.sum())
        residual = float(params_by_sku[row.sku_id]['residual_std'])
        safety = 1.65 * residual * np.sqrt(lead)
        target = lead_demand + safety
        inventory.loc[i, 'recommended_order_units'] = max(0, int(np.ceil(target-row.starting_on_hand_units)))
        inventory.loc[i, 'risk_value_estimate'] = round(float(row.units_expiring_unconsumed_12w * p.unit_cost), 2)
    inventory = inventory.merge(products[['sku_id','category','unit_cost','lead_time_weeks']], on='sku_id', how='left')
    inventory.to_csv(OUT / 'expiry_risk_and_reorder.csv', index=False)
    save_model(params_by_sku, MODEL, {'algorithm':'Per-SKU ridge regression', 'ridge_alpha':8.0,
                'features':['trend','annual sine','annual cosine','lag 1','lag 4','lag 13','lag 52'],
                'training_weeks_per_sku':int(sales.week_start.nunique()), 'forecast_horizon_weeks':HORIZON,
                'synthetic_only':True})
    return summary


if __name__ == '__main__':
    print(json.dumps(run_pipeline(), indent=2))



