from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SEED = 24017
N_SKUS = 24
N_WEEKS = 104


def build_dataset(seed: int = SEED) -> None:
    """Create fictional weekly sales, product, and lot data. No real pharmacy data."""
    rng = np.random.default_rng(seed)
    DATA.mkdir(parents=True, exist_ok=True)
    specs = []
    categories = [('Analgesic', 8), ('Cold and Flu', 6), ('Chronic Care', 6), ('Gastrointestinal', 4)]
    n = 1
    for category, count in categories:
        for _ in range(count):
            base = float(rng.uniform(18, 85))
            trend = float(rng.uniform(-0.0004, 0.0008))
            amp = float(rng.uniform(0.08, 0.30)) if category == 'Cold and Flu' else float(rng.uniform(0.01, 0.10))
            phase = float(rng.uniform(-0.25, 0.25))
            cost = float(rng.uniform(1.2, 18.0))
            specs.append({'sku_id': f'MED{n:03d}', 'category': category,
                          'base_demand': base, 'trend': trend, 'season_amp': amp,
                          'season_phase': phase, 'unit_cost': round(cost, 2),
                          'lead_time_weeks': int(rng.integers(1, 4))})
            n += 1
    products = pd.DataFrame([{k: v for k, v in s.items() if k not in ('base_demand','trend','season_amp','season_phase')} for s in specs])
    products.to_csv(DATA / 'products.csv', index=False)

    dates = pd.date_range('2024-10-28', periods=N_WEEKS, freq='W-MON')
    weekly = []
    for s in specs:
        for t, dt in enumerate(dates):
            annual = np.sin(2 * np.pi * (t / 52.18) + s['season_phase'])
            winter_bump = 0.20 if s['category'] == 'Cold and Flu' and dt.month in (12, 1, 2) else 0.0
            expected = s['base_demand'] * (1 + s['trend'] * t) * (1 + s['season_amp'] * annual + winter_bump)
            expected = max(1.0, expected)
            units = int(rng.poisson(expected))
            weekly.append({'week_start': dt.date().isoformat(), 'sku_id': s['sku_id'],
                           'category': s['category'], 'demand_units': units,
                           'promotion_flag': int(rng.random() < 0.08)})
    sales = pd.DataFrame(weekly)
    sales.to_csv(DATA / 'weekly_demand.csv', index=False)

    snapshot = dates[-1] + pd.Timedelta(days=7)
    lots = []
    for s in specs:
        recent_mean = sales.loc[sales.sku_id == s['sku_id'], 'demand_units'].tail(8).mean()
        on_hand = max(8, int(rng.normal(recent_mean * 7.5, recent_mean * 1.5)))
        # Mix expiry horizons to create realistic variation in short-dated inventory.
        shares = rng.dirichlet([1.5, 2.0, 2.5, 3.5])
        quantities = np.floor(shares * on_hand).astype(int)
        quantities[-1] += on_hand - int(quantities.sum())
        for j, (weeks_out, qty) in enumerate(zip([2, 5, 9, 18], quantities), start=1):
            lots.append({'snapshot_date': snapshot.date().isoformat(), 'sku_id': s['sku_id'],
                         'lot_id': f'{s["sku_id"]}-LOT{j}', 'on_hand_units': int(qty),
                         'expiry_date': (snapshot + pd.Timedelta(weeks=int(weeks_out))).date().isoformat()})
    pd.DataFrame(lots).to_csv(DATA / 'inventory_lots.csv', index=False)


if __name__ == '__main__':
    build_dataset()
    print(f'Wrote synthetic data to {DATA}')

