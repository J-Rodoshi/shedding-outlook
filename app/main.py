"""Backend for real PGCB-derived out-of-sample load-shedding forecasts.

The website uses historical out-of-sample predictions, not live estimates.
An additional endpoint accepts an explicitly supplied 28-feature input vector
and scores it with separately labelled, final-refit XGBoost models.
"""
from __future__ import annotations

import json
import math
import atexit
from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
OUTLOOK = ROOT / 'web/data/outlook.json'
MANIFEST = ROOT / 'artifacts/models/manifest.json'
LOCATIONS = {
    'bangladesh': 'All Bangladesh',
    'barishal': 'Barishal Division',
    'chattogram': 'Chattogram Division',
    'dhaka': 'Dhaka Division',
    'khulna': 'Khulna Division',
    'mymensingh': 'Mymensingh Division',
    'rajshahi': 'Rajshahi Division',
    'rangpur': 'Rangpur Division',
    'sylhet': 'Sylhet Division',
}

app = FastAPI(title='Shedding Outlook', version='1.0.0',
              description='Bangladesh national load-shedding outlook from archived out-of-sample predictions.')


@lru_cache(maxsize=1)
def dataset():
    if not OUTLOOK.exists():
        raise RuntimeError('Export original notebook predictions with scripts/export_outlook.py first')
    data = json.loads(OUTLOOK.read_text(encoding='utf-8'))
    if data.get('meta', {}).get('synthetic') is not False:
        raise RuntimeError('Synthetic prediction data is not accepted by the deployed API')
    if not data.get('rows'):
        raise RuntimeError('The historical prediction file is empty')
    return data


@lru_cache(maxsize=1)
def model_manifest():
    if not MANIFEST.exists():
        raise RuntimeError('No final-refit model artifacts available')
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


@app.get('/health')
def health():
    d = dataset()
    return {'status': 'ok', 'mode': d['meta']['mode'],
            'synthetic': False, 'first_hour': d['meta']['first_hour'],
            'last_hour': d['meta']['last_hour'],
            'n_origin_hours': d['meta']['n_origin_hours'],
            'model_artifacts': MANIFEST.exists()}


@app.get('/api/meta')
def meta():
    return {'data': dataset()['meta'], 'locations': LOCATIONS,
            'geographic_scope': 'National Bangladesh only; divisions do not change the forecast'}


@app.get('/api/hour')
def hour(
    date: Date,
    hour: Annotated[int, Query(ge=0, le=23)] = 12,
    location: str = 'bangladesh',
):
    if location not in LOCATIONS:
        raise HTTPException(status_code=422, detail='Select a supported Bangladesh location')
    key = f'{date.isoformat()}T{hour:02d}'
    d = dataset()
    r = d['rows'].get(key)
    if r is None:
        raise HTTPException(status_code=404, detail='No evaluated prediction for this hour')
    forecast = {}
    for idx, h in enumerate((1, 6, 24)):
        richer = r[1 + idx]
        base = r[4 + idx]
        chosen = richer if richer is not None else base
        forecast[str(h)] = {
            'probability': chosen,
            'expanded_probability': richer,
            'baseline_probability': base,
            'source_model': 'expanded' if richer is not None else ('baseline' if base is not None else None),
            'observed_outcome': d.get('observed_targets', {}).get(key, [None] * 3)[idx],
        }
    return {'origin': key + ':00', 'timezone': 'Asia/Dhaka',
            'selected_location': LOCATIONS[location],
            'scope': 'Bangladesh national aggregate',
            'now_shedding': r[0], 'horizons': forecast,
            'mode': 'historical_out_of_sample'}


class FeatureRequest(BaseModel):
    origin: str = Field(description='Origin time (local Asia/Dhaka; label only)')
    features: dict[str, float]


@lru_cache(maxsize=18)
def get_models(name: str, horizon: int):
    from xgboost import XGBClassifier
    m = model_manifest()
    key = f'{name}_{horizon}h'
    paths = m['files'].get(key)
    if not paths:
        raise ValueError(f'Unrecognised model {key}')
    out = []
    for p in paths:
        model = XGBClassifier()
        model.load_model(ROOT / 'artifacts/models' / p)
        out.append(model)
    return out


@app.post('/api/predict-from-features')
def predict_features(request: FeatureRequest):
    """True on-demand inference, ONLY when a caller supplies all origin features.

    Unlike the archived prediction API, this scores final-refit models. This
    endpoint does not claim that the supplied features come from a live feed.
    """
    import pandas as pd
    m = model_manifest()
    full = m['features']['expanded']
    missing = [name for name in full if name not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail={'missing_features': missing})
    vals = [request.features[k] for k in full]
    if any(not math.isfinite(float(v)) for v in vals):
        raise HTTPException(status_code=422, detail='All feature values must be finite')
    result = {}
    for h in m['horizons_hours']:
        result[str(h)] = {}
        for name in ('expanded', 'baseline'):
            columns = m['features'][name]
            x = pd.DataFrame([{f: request.features[f] for f in columns}])[columns]
            models = get_models(name, h)
            result[str(h)][name] = round(
                sum(float(model.predict_proba(x)[:, 1][0]) for model in models) / len(models),
                6,
            )
    return {'mode': 'final_refit_inference_from_supplied_features',
            'origin': request.origin, 'scope': 'Bangladesh national aggregate',
            'probabilities': result,
            'warning': 'Input features are user supplied; no live PGCB feed is connected.'}


# Release native XGBoost objects before Python unloads the XGBoost library.
atexit.register(get_models.cache_clear)

# Keep this mount LAST so /api and /health take precedence.
app.mount('/', StaticFiles(directory=str(ROOT / 'web'), html=True), name='web')
