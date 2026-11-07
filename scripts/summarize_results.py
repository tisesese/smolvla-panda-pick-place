"""Recompute paired-trial metrics with Python's standard library only."""
import argparse
from collections import defaultdict
from pathlib import Path
import csv
import json
from statistics import mean
from decimal import Decimal,ROUND_HALF_UP

ROOT=Path(__file__).resolve().parents[1]

def summarize(check=False):
    rows=[json.loads(x) for x in (ROOT/'results/results.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    groups=defaultdict(list)
    for row in rows:groups[row['model_condition'],row['lighting']].append(row)
    if check:
        assert len(rows)==200 and len(groups)==4
        reference=None
        for key,items in groups.items():
            assert len(items)==50 and {x['index'] for x in items}==set(range(50)),key
            assert sum(x['category']=='interior' for x in items)==40
            assert sum(x['category']=='boundary' for x in items)==10
            paired={x['index']:(x['seed'],x['position']['xyz_m']) for x in items}
            if reference is None:reference=paired
            else:assert paired==reference,'Position pairing changed'
    summary=[]
    for model,light in [('finetuned33000','normal'),('base','normal'),('finetuned33000','dim'),('base','dim')]:
        items=groups[model,light]
        for category in ['all','interior','boundary']:
            if model=='base' and category!='all':continue
            sample=items if category=='all' else [x for x in items if x['category']==category]
            successes=[x for x in sample if x['success']]
            # Simulation timestamps accumulate tiny floating-point errors.
            # Round far below the 2ms physics step before reporting means.
            duration=sum((Decimal(str(round(x['sim_seconds'],6))) for x in successes),Decimal(0))/len(successes) if successes else None
            summary.append(dict(model=model,lighting=light,category=category,attempts=len(sample),successes=len(successes),success_rate=len(successes)/len(sample),mean_success_sim_s=float(duration) if duration is not None else None))
    if check:
        observed={(x['model'],x['lighting']):x['successes'] for x in summary if x['category']=='all'}
        assert observed=={('finetuned33000','normal'):45,('base','normal'):0,('finetuned33000','dim'):46,('base','dim'):0},observed
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--write',action='store_true');args=parser.parse_args()
    summary=summarize(args.check)
    if args.write:
        with (ROOT/'results/summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        (ROOT/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in summary:
        duration='N/A' if r['mean_success_sim_s'] is None else str(Decimal(str(r['mean_success_sim_s'])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'s'
        print(f"{r['model']:18} {r['lighting']:6} {r['category']:8} {r['successes']}/{r['attempts']}  {100*r['success_rate']:.0f}%  {duration}")
