import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app, dataset, LOCATIONS, ROOT, model_manifest

client = TestClient(app)


def test_health_real_data():
    res=client.get('/health');assert res.status_code==200
    body=res.json();assert body['status']=='ok' and body['synthetic'] is False
    assert body['n_origin_hours']>=15000
    assert body['mode']=='out_of_sample_rolling_origin'


def test_home_page_and_scripts():
    x=client.get('/');assert x.status_code==200 and 'Shedding Outlook' in x.text
    assert 'Location' in x.text and 'Typical' in x.text and 'Analyst' in x.text
    assert client.get('/static/app.js').status_code==200


def test_meta_reports_national_scope():
    x=client.get('/api/meta');assert x.status_code==200
    m=x.json();assert len(m['locations'])==9
    assert 'National' in m['geographic_scope']


def test_outlook_uses_actual_saved_forecast():
    data=dataset()
    first=next(k for k,v in data['rows'].items() if v[2] is not None)
    x=client.get('/api/hour',params={'date':first[:10],'hour':int(first[11:13])})
    assert x.status_code==200
    o=x.json();assert o['mode']=='historical_out_of_sample'
    assert o['now_shedding'] in (0,1)
    assert o['horizons']['6']['expanded_probability']==data['rows'][first][2]
    assert o['horizons']['6']['source_model']=='expanded'


def test_location_does_not_fabricate_division_prediction():
    k=dataset()['meta']['first_hour']
    opts={'date':k[:10],'hour':int(k[11:13])}
    a=client.get('/api/hour',params={**opts,'location':'bangladesh'}).json()
    b=client.get('/api/hour',params={**opts,'location':'dhaka'}).json()
    assert a['horizons']==b['horizons']
    assert 'national' in b['scope'].lower()
    assert b['selected_location']=='Dhaka Division'


def test_invalid_location_rejected():
    assert client.get('/api/hour',params={'date':'2024-02-01','hour':5,'location':'usa'}).status_code==422


def test_invalid_hour_rejected():
    assert client.get('/api/hour',params={'date':'2024-02-01','hour':24}).status_code==422


def test_missing_date_not_fabricated():
    x=client.get('/api/hour',params={'date':'2026-11-23','hour':12})
    assert x.status_code==404


def test_metadata_and_targets_are_consistent():
    d=dataset()
    assert d['meta']['synthetic'] is False
    assert len(d['rows']) == len(d['observed_targets']) == d['meta']['n_origin_hours']
    for k in list(d['rows'])[:100]:
        assert len(d['rows'][k])==7
        assert len(d['observed_targets'][k])==3
        assert all(x is None or x in (0,1) for x in d['observed_targets'][k])


def test_report_json_original_metrics():
    r=client.get('/data/report.json');assert r.status_code==200
    d=r.json()
    by_h={x['horizon']:x for x in d['brier']}
    assert by_h[6]['baseline']==0.14
    assert by_h[6]['expanded']==0.123
    assert by_h[24]['gain']=='Not established'


def test_model_manifest_has_all_18_models():
    m=model_manifest()
    assert len(m['files'])==6
    assert all(len(v)==3 for v in m['files'].values())
    assert all((ROOT/'artifacts/models'/name).exists() for paths in m['files'].values() for name in paths)


def test_predict_requires_full_original_feature_vector():
    r=client.post('/api/predict-from-features',json={'origin':'2025-07-15T10:00','features':{'hour':10}})
    assert r.status_code==422
    assert 'missing_features' in r.json()['detail']


def test_predict_returns_three_horizons_for_complete_features():
    m=model_manifest();features={f:0.0 for f in m['features']['expanded']}
    x=client.post('/api/predict-from-features',json={'origin':'2026-03-08T10:00','features':features})
    assert x.status_code==200,x.text
    d=x.json();assert d['mode']=='final_refit_inference_from_supplied_features'
    assert set(d['probabilities'])=={'1','6','24'}
    for h in d['probabilities']:
        assert all(0<=p<=1 for p in d['probabilities'][h].values())
    assert 'no live PGCB' in d['warning']
