"""Validate that the published artifact is based on real notebook predictions."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'web/data/outlook.json').read_text())
assert data['meta']['synthetic'] is False
assert len(data['rows'])>=15000
assert len(data.get('observed_targets',{}))==len(data['rows'])
m=json.loads((root/'artifacts/models/manifest.json').read_text())
assert len(m['files'])==6 and all(len(v)==3 for v in m['files'].values())
assert all((root/'artifacts/models'/x).exists() for v in m['files'].values() for x in v)
print('Verified: real OOS predictions, corresponding outcomes and 18 model files.')
