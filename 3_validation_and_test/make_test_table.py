"""
Main test-set table: the final classifier (majority vote of the 40 fixed classifiers)
with and without SOM features, and their paired difference with a bootstrap 95% CI.

    python make_test_table.py        -> paper_figs/test_final_model/test_main_table.{csv,tex}

From results_final_sall (no refitting). PR-AUC uses the score averaged over the
40 seeds. Paired bootstrap: the 7,135 test objects are resampled with replacement;
both pipelines are scored on the same resample. The 48 objects without a
light-curve grid count as non-TDE with score 0.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ARMS = ['Without SOM', 'With SOM']
B = 10_000


def metrics(y, pred, score):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r,
            'F1': 2 * p * r / (p + r), 'PR-AUC': average_precision_score(y, score)}


def main():
    p = pd.read_csv('results_final_sall/tables/test_predictions.csv.gz')
    lab = pd.read_csv('public_test_log.csv').set_index('object_id').SpecType
    y = (lab == 'TDE').to_numpy()
    pred, score = {}, {}
    for a in ARMS:
        g = p[p.arm == a]
        pred[a] = (g.pivot(index='object_id', columns='repeat', values='votes')
                   .reindex(lab.index).fillna(0).mean(1) >= .5).to_numpy()
        score[a] = (g.pivot(index='object_id', columns='repeat', values='score')
                    .reindex(lab.index).fillna(0).mean(1)).to_numpy()
    rows = {a: metrics(y, pred[a], score[a]) for a in ARMS}
    diff = {k: rows['With SOM'][k] - rows['Without SOM'][k] for k in rows['With SOM']}

    rng = np.random.default_rng(0)
    boot = {k: [] for k in diff}
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        m = {a: metrics(y[i], pred[a][i], score[a][i]) for a in ARMS}
        for k in diff:
            boot[k].append(m['With SOM'][k] - m['Without SOM'][k])
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}
    pos = {k: np.mean(np.asarray(v) > 0) for k, v in boot.items()}

    t = pd.DataFrame(rows).T
    t.loc['With - Without'] = pd.Series(diff)
    t.loc['95% CI low'] = pd.Series({k: v[0] for k, v in ci.items()})
    t.loc['95% CI high'] = pd.Series({k: v[1] for k, v in ci.items()})
    t.loc['P(diff > 0)'] = pd.Series(pos)
    out = 'paper_figs/test_final_model/test_main_table'
    t.to_csv(f'{out}.csv')
    pd.set_option('display.width', 200)
    print(t.round(4).to_string())

    def f(k, v):
        return f'{v:.0f}' if k in ('TP', 'FP', 'FN') else f'{v:.3f}'

    def fd(k):
        s = '+' if diff[k] >= 0 else '$-$'
        lo, hi = ci[k]
        fmt = (lambda x: f'{x:+.0f}') if k in ('TP', 'FP', 'FN') else (lambda x: f'{x:+.3f}')
        return f'{s}{abs(diff[k]):.0f}' if k in ('TP', 'FP', 'FN') else f'{s}{abs(diff[k]):.3f}', \
               f'[{fmt(lo)}, {fmt(hi)}]'.replace('-', '$-$')

    cols = ['TP', 'FP', 'FN', 'precision', 'recall', 'F1', 'PR-AUC']
    lines = [' & '.join(['Without SOM'] + [f(k, rows['Without SOM'][k]) for k in cols]) + r' \\',
             ' & '.join(['With SOM'] + [f(k, rows['With SOM'][k]) for k in cols]) + r' \\',
             r'\hline',
             ' & '.join(['Difference'] + [fd(k)[0] for k in cols]) + r' \\',
             ' & '.join(['95\\% CI'] + [fd(k)[1] for k in cols]) + r' \\']
    open(f'{out}.tex', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}.csv / .tex')


if __name__ == '__main__':
    main()
