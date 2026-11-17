"""Portable evaluator for data/examples/mini_uniform and output/mini_uniform_run_1.

Derived demonstration utility. See the original evaluator for the historical analyses.
Run from repository root: python benchmarks/evaluators/evaluate_mini_uniform.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / 'data/examples/mini_uniform/true_vector_values.npz'
RUN = ROOT / 'output/mini_uniform_run_1'
OUT = RUN / 'mini_validation'


def main():
    if not GT.is_file():
        raise FileNotFoundError(f'{GT} (run mini generator first)')
    paths = sorted((RUN/'validation').glob('validation_*.npz'))
    if not paths:
        raise FileNotFoundError(f'No validation files in {RUN / "validation"}; run pipeline first')
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(GT) as d:
        ut=d['u_gt'];vt=d['v_gt']
    records=[]
    for path in paths:
        with np.load(path) as a:
            x=a['x'].astype(int);y=a['y'].astype(int)
            valid=a['valid_mask'].astype(bool)
            delta=np.hypot(a['u_output']-ut[y,x],a['v_output']-vt[y,x])
            raw=np.hypot(a['u_raw']-ut[y,x],a['v_raw']-vt[y,x])
            record={
                'pair':int(path.stem.rsplit('_',1)[-1]),
                'total_sampled':valid.size,
                'retained':int(np.count_nonzero(valid)),
                'retention_percent':float(100*np.mean(valid)),
                'raw_median_epe_px':float(np.median(raw)),
                'final_median_epe_retained_px':float(np.median(delta[valid])) if valid.any() else np.nan,
                'final_p95_epe_retained_px':float(np.percentile(delta[valid],95)) if valid.any() else np.nan,
                'mean_u_retained':float(np.mean(a['u_output'][valid])) if valid.any() else np.nan,
                'mean_v_retained':float(np.mean(a['v_output'][valid])) if valid.any() else np.nan,
            }
        records.append(record)
    with (OUT/'mini_uniform_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    pairs=[r['pair'] for r in records]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(pairs,[r['raw_median_epe_px'] for r in records],label='raw, all sampled')
    ax.plot(pairs,[r['final_median_epe_retained_px'] for r in records],label='retained output')
    ax.set(xlabel='Frame pair',ylabel='Median endpoint error [px/frame]',title='New mini uniform test (u = 1, v = 0)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'mini_epe.png',dpi=180);plt.close(fig)
    print('Created',OUT,'from',len(records),'frame pairs')
    print('Median retained EPE over pairs:',float(np.nanmedian([r['final_median_epe_retained_px'] for r in records])))
    print('Median retention over pairs:',float(np.nanmedian([r['retention_percent'] for r in records])))


if __name__=='__main__':
    main()
