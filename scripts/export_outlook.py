"""Export *real* out-of-sample forecasts from the AIT load-shedding notebook.

Uses the same 12-feature baseline, exploratory 28-feature model, XGBoost
hyperparameters, seed ensemble (0/1/2), horizon purge gap and nine quarterly
expanding-window test folds as the manuscript's final comparison.

Usage from repository root:
    python scripts/export_outlook.py --data "C:/.../CP Electricity Project"

The exporter DOES NOT use any internet source, simulate predictions, or
claim to forecast today's conditions without current PGCB measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HORIZONS = (1, 6, 24)
SEEDS = (0, 1, 2)
PERSISTENCE = [
    "load_shedding_lag1", "load_shedding_lag24", "load_shedding_lag168",
    "load_shedding_roll24", "load_shedding_roll168",
    "load_shedding_roll24_count", "load_shedding_roll168_count",
]
BASELINE = PERSISTENCE + ["gap_now", "shed_now", "hour", "hour_sin", "hour_cos"]
EXPANDED = [
    "dayofweek", "demand_mw", "demand_mw_lag1", "demand_mw_lag24", "gap_now",
    "generation_mw", "generation_mw_lag1", "generation_mw_lag24", "hour",
    "hour_cos", "hour_sin", "india_bheramara_hvdc", "india_tripura",
    "is_holiday", "is_weekend", "load_shedding_lag1",
    "load_shedding_lag168", "load_shedding_lag24",
    "load_shedding_roll168", "load_shedding_roll168_count",
    "load_shedding_roll24", "load_shedding_roll24_count",
    "national_humidity", "national_precipitation", "national_solar_radiation",
    "national_temperature", "national_wind_speed", "shed_now",
]
assert len(BASELINE) == 12 and len(EXPANDED) == 28


def load_table(path):
    """Read original parquet, or a numerically equivalent CSV for portability."""
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            if not path.with_suffix('.csv').exists():
                raise RuntimeError('Install pyarrow to read Parquet: pip install pyarrow')
    csv = path.with_suffix('.csv')
    if not csv.exists():
        raise FileNotFoundError(f'Cannot find {path} or {csv}')
    return pd.read_csv(csv)


def data_and_labels(data):
    source = Path(data)
    x = load_table(source / 'modeling_ready.parquet').copy()
    reg = load_table(source / 'checkpoint_7_regularized.parquet').copy()
    for d in [x, reg]:
        d['datetime'] = pd.to_datetime(d['datetime'])
        if d['datetime'].dt.tz is not None:
            raise ValueError('Expected naive local Asia/Dhaka timestamps')
        if d['datetime'].duplicated().any():
            raise ValueError('Duplicate hourly timestamps')
    x.sort_values('datetime', inplace=True)
    x['gap_now'] = x['demand_mw'] - x['generation_mw']
    x['shed_now'] = (x['load_shedding'] > 0).astype('int8')
    reg.sort_values('datetime', inplace=True)
    ser = reg.set_index('datetime')['load_shedding']
    missing = set(BASELINE + EXPANDED) - set(x.columns)
    if missing:
        raise KeyError(f'Missing model features: {sorted(missing)}')
    if x[BASELINE + EXPANDED].isna().any().any():
        raise ValueError('Unexpected missing model inputs. Recheck notebook preprocessing.')
    return x, ser


def training_frame(x, ser, h):
    valid_times = pd.DatetimeIndex(x['datetime'] + pd.Timedelta(hours=h))
    target_mw = ser.reindex(valid_times).to_numpy()
    present = ~pd.isna(target_mw)
    a = x.loc[present].copy()
    a['y'] = (target_mw[present] > 0).astype('int8')
    return a


def fit(train, features, seed, n_jobs):
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=seed, n_jobs=n_jobs,
    )
    return clf.fit(train[features], train['y'])


def calculate_metrics(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def export(args):
    x, ser = data_and_labels(args.data)
    groups = {'expanded': EXPANDED, 'baseline': BASELINE}
    rows = {}
    actuals = {}
    pooled = {(h, model): [] for h in HORIZONS for model in groups}
    pooled_y = {h: [] for h in HORIZONS}
    folds = []
    quarters = pd.date_range('2024-01-01', '2026-04-01', freq='QS')
    for h_idx, h in enumerate(HORIZONS):
        a = training_frame(x, ser, h)
        gap = pd.Timedelta(hours=h)
        for start, end in zip(quarters[:-1], quarters[1:]):
            tr = a[a['datetime'] < start - gap]
            te = a[(a['datetime'] >= start) & (a['datetime'] < end)]
            if te['y'].nunique() < 2:
                continue  # match the notebook's fold eligibility rule
            if len(tr) == 0 or tr['y'].nunique() < 2:
                raise ValueError(f'Invalid training fold: {h}h, {start}')
            pooled_y[h].append(te['y'].to_numpy(dtype=int))
            for model_idx, (name, features) in enumerate(groups.items()):
                ensemble = []
                for seed in SEEDS:
                    model = fit(tr, features, seed, args.jobs)
                    ensemble.append(model.predict_proba(te[features])[:, 1])
                pred = np.mean(np.stack(ensemble), axis=0)
                pooled[h, name].append(pred)
                folds.append({
                    'horizon_hours': h, 'quarter_start': str(start.date()),
                    'model': name, 'n': len(te),
                    'brier': calculate_metrics(te['y'].values, pred),
                })
                for ts, shed, y_true, probability in zip(
                    te['datetime'], te['shed_now'], te['y'], pred
                ):
                    key = ts.strftime('%Y-%m-%dT%H')
                    state = rows.setdefault(key, [None] * 7)
                    state[0] = int(shed)
                    state[1 + h_idx + (3 if name == 'baseline' else 0)] = round(float(probability), 6)
                    truth = actuals.setdefault(key, [None, None, None])
                    truth[h_idx] = int(y_true)
            print(f'{h:2}h, {start:%Y-%m}: {len(te):>5} origins  |  '
                  f'expanded={folds[-2]["brier"]:.4f}, baseline={folds[-1]["brier"]:.4f}', flush=True)
    totals = []
    for h in HORIZONS:
        y = np.concatenate(pooled_y[h])
        for name in groups:
            pred = np.concatenate(pooled[h, name])
            totals.append({'horizon_hours': h, 'model': name, 'n': len(y),
                           'brier': calculate_metrics(y, pred)})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(rows)
    if not keys:
        raise RuntimeError('No historical predictions were generated')
    payload = {
        'meta': {
            'synthetic': False,
            'mode': 'out_of_sample_rolling_origin',
            'location': 'Bangladesh (national aggregate)',
            'first_hour': keys[0], 'last_hour': keys[-1],
            'timezone': 'Asia/Dhaka', 'horizons_hours': list(HORIZONS),
            'models': ['expanded_28_features', 'state_aware_baseline_12_features'],
            'seeds': list(SEEDS), 'n_origin_hours': len(keys),
            'training_protocol': 'Nine expanding-quarter folds, target gap h',
            'data_last_hour': str(x['datetime'].max()),
            'note': 'Historical out-of-sample predictions, not live observations',
        },
        'rows': {k: rows[k] for k in keys},
        'observed_targets': {k: actuals[k] for k in keys},
    }
    out_path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    metrics_path = out_path.with_name('export_validation.csv')
    pd.DataFrame(totals).to_csv(metrics_path, index=False)
    pd.DataFrame(folds).to_csv(out_path.with_name('export_fold_validation.csv'), index=False)
    print(f'Exported {len(keys):,} real origin hours to {out_path}', flush=True)
    print(pd.DataFrame(totals).to_string(index=False), flush=True)

    reference = Path(args.data) / 'results/final_summary.csv'
    if reference.exists():
        saved = pd.read_csv(reference)
        worst = 0
        for r in totals:
            model_name = 'frozen (exploratory)' if r['model'] == 'expanded' else 'pers+state+hour'
            official = saved.loc[(saved.horizon_hours == r['horizon_hours']) & (saved.model == model_name), 'Brier_pooled']
            if len(official) != 1:
                raise AssertionError(f'Missing official reference: {r}')
            diff = abs(float(official.iloc[0]) - r['brier'])
            worst = max(diff, worst)
            if diff > 0.001:
                raise AssertionError(f'Export Brier does not match notebook within 0.001: {r}, official={official.iloc[0]}')
        print(f'Validated against notebook final_summary.csv (max Brier difference {worst:.6f})', flush=True)

    if not args.skip_models:
        model_dir = Path(args.models)
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = {'model_type': 'xgboost', 'mode': 'final_refit',
                    'not_rolling_origin_test_models': True,
                    'timezone': 'Asia/Dhaka', 'seeds': list(SEEDS), 'horizons_hours': list(HORIZONS),
                    'features': groups, 'files': {},
                    'training_end': str(x['datetime'].max())}
        for h in HORIZONS:
            a = training_frame(x, ser, h)
            for name, features in groups.items():
                paths = []
                for seed in SEEDS:
                    model = fit(a, features, seed, args.jobs)
                    fname = f'{name}_{h}h_seed{seed}.json'
                    model.save_model(model_dir / fname)
                    paths.append(fname)
                manifest['files'][f'{name}_{h}h'] = paths
                print('Refit model:', name, h, 'hours', flush=True)
        (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='.', help='folder with notebook Parquet files')
    ap.add_argument('--out', default='web/data/outlook.json')
    ap.add_argument('--models', default='artifacts/models')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--skip-models', action='store_true')
    export(ap.parse_args())

if __name__ == '__main__':
    main()
