"""Convert original research result tables into website metrics (no new training)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

LABELS = {1:'Next hour',6:'Next 6 hours',24:'Tomorrow'}
CALIB = {1:'Small gain. Probability calibration is comparatively reliable here.',
         6:'Clearest improvement. Treat the forecast as a risk ranking rather than a precise chance.',
         24:'The difference is within the uncertainty range; improvement is not established.'}
GRP={'Load-shedding history':'Recent shedding history','Demand-generation':'Demand and generation','Weather':'Weather','Calendar':'Calendar','Cross-border':'Cross-border imports'}

def build(src, dst):
    root=Path(src)
    summary=pd.read_csv(root/'results/final_summary.csv')
    folds=pd.read_csv(root/'results/fold_summary.csv')
    groups=pd.read_csv(root/'results/grouped_permutation_importance.csv')
    brier=[]
    for h in (1,6,24):
        q=summary[summary.horizon_hours==h]
        base=float(q[q.model=='pers+state+hour']['Brier_pooled'].iloc[0])
        full=float(q[q.model=='frozen (exploratory)']['Brier_pooled'].iloc[0])
        ci_high=float(q[q.model=='frozen (exploratory)']['Brier_diff_CI_high'].iloc[0])
        improvement=(base-full)/base*100
        gain=(f'{improvement:.1f}% better' if ci_high<0 else 'Not established')
        brier.append({'horizon':h,'label':LABELS[h],'baseline':base,'expanded':full,
                      'gain':gain,'interpretation':CALIB[h]})
    imp=groups[groups.h==6].copy()
    importance=[[GRP[row.group],float(row.brier_increase)] for _,row in imp.iterrows()]
    importance.sort(key=lambda x:-x[1])
    qtr=[]
    for start in sorted(set(folds[(folds.horizon_hours==6)].period_start)):
        rr=folds[(folds.horizon_hours==6)&(folds.period_start==start)]
        base=float(rr[rr.model=='pers+state+hour']['Brier'].iloc[0]);full=float(rr[rr.model=='frozen (exploratory)']['Brier'].iloc[0])
        qtr.append([f'{start[:4]} Q{(int(start[5:7])-1)//3+1}',round(full-base,4)])
    report={
        'source':'Original project notebook results CSVs',
        'typical':[
            ['Hot weather', [['Coldest tenth of hours',9.7],['Hottest tenth of hours',68.8]]],
            ['Holidays', [['Public holidays',26.7],['Other days',42.6]]],
            ['Year by year', [['2022',26.1],['2023',57.1],['2024',57.3],['2025',37.6]]],
        ],'brier':brier,'importance':importance,'quarters':qtr,
    }
    out=Path(dst);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Wrote',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',default='web/data/report.json');a=ap.parse_args();build(a.data,a.out)
