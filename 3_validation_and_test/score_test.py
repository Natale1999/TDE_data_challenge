"""
Score the external test set. The ONLY script that reads the test labels.

    python score_test.py results_wincol_test            (reads <prefix>_s1, <prefix>_s21)

Input: <prefix>_s*/tables/test_predictions.csv.gz from cv_som.py --test (no labels).
Every fold model of every repetition scored every test object; per repetition an
object is TDE if most of its 5 fold models say so, and its score is their mean.
Objects without a light-curve grid (no SOM input) count as not TDE and rejected
at stage 1, so every metric is over all objects of public_test_log.csv.

Outputs
    <prefix>_pooled/tables/test_per_repeat.csv     metrics per repetition and arm
    <prefix>_pooled/tables/test_summary.csv        mean +/- sd, 40-repetition vote
    paper_figs/<FIGDIR>/test_cm_no_som, test_cm_som stage 1 (top) + final (bottom),
                                                    mean +/- sd over repetitions
    paper_figs/<FIGDIR>/test_ensemble_cm            vote of all repetitions
    FIGDIR = 2nd argument, default test_final_model (fold-model run: test_fold_models)
    printed: summary, paired With - Without SOM, comparison table
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from paper_figures import draw_cm, save

PRE = sys.argv[1] if len(sys.argv) > 1 else 'results_wincol_test'
FIGDIR = sys.argv[2] if len(sys.argv) > 2 else 'test_final_model'   # subfolder of paper_figs/
OUT = f'{PRE}_pooled'
LABELS = 'public_test_log.csv'
ARMS = ['Without SOM', 'With SOM']


def load():
    files = sorted(glob.glob(f'{PRE}_s*/tables/test_predictions.csv.gz'))
    if not files:
        sys.exit(f'no {PRE}_s*/tables/test_predictions.csv.gz')
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lab = pd.read_csv(LABELS).set_index('object_id').SpecType
    print(f'{len(files)} run(s), {p.repeat.nunique()} repetitions, {p.arm.nunique()} arms, '
          f'{p.object_id.nunique()} of {len(lab)} test objects with a light-curve grid')
    missing = lab.index.difference(p.object_id.unique())
    print(f'  {len(missing)} objects without a grid -> not TDE '
          f'({(lab.loc[missing] == "TDE").sum()} TDEs, {(lab.loc[missing] == "AGN").sum()} AGN among them)')
    return p, lab


def per_repeat(p, lab):
    ids = lab.index
    y = (lab == 'TDE').to_numpy()
    agn = (lab == 'AGN').to_numpy()
    rows, calls, scores, s1 = [], {}, {}, {}
    for (arm, rep), g in p.groupby(['arm', 'repeat']):
        g = g.set_index('object_id').reindex(ids)
        n = g.n_folds.max()
        pred = (g.votes.fillna(0) * 2 > n).to_numpy()
        score = g.score.fillna(0).to_numpy()
        passed = (g.s1_votes.fillna(0) * 2 > n).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({'arm': arm, 'repeat': rep, 'precision': tp / max(tp + fp, 1),
                     'recall': tp / (tp + fn), 'F1': f1_score(y, pred),
                     'PR-AUC': average_precision_score(y, score),
                     'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                     'FP_AGN': int((pred & agn).sum()), 'FP_SN': int((pred & ~agn & ~y).sum()),
                     's1_agn_rejected': int((agn & ~passed).sum()),
                     's1_agn_passed': int((agn & passed).sum()),
                     's1_tr_rejected': int((~agn & ~passed).sum()),
                     's1_tr_passed': int((~agn & passed).sum()),
                     's1_tde_lost': int((y & ~passed).sum())})
        calls.setdefault(arm, {})[rep] = pred
        scores.setdefault(arm, {})[rep] = score
        s1.setdefault(arm, {})[rep] = passed
    return pd.DataFrame(rows), calls, scores, s1, y, agn


def ms(x):
    return f'{x.mean():.3f} +/- {x.std():.3f}'


def main():
    os.makedirs(f'{OUT}/tables', exist_ok=True)
    os.makedirs(f'paper_figs/{FIGDIR}', exist_ok=True)
    p, lab = load()
    r, calls, scores, s1, y, agn = per_repeat(p, lab)
    r.to_csv(f'{OUT}/tables/test_per_repeat.csv', index=False)
    R = r.repeat.nunique()

    # ---- summary per arm: per repetition (mean +/- sd) and the vote of all repetitions
    summ = []
    for arm, g in r.groupby('arm'):
        V = np.mean([calls[arm][k] for k in sorted(calls[arm])], axis=0) >= .5
        S = np.mean([scores[arm][k] for k in sorted(scores[arm])], axis=0)
        tn, fp, fn, tp = confusion_matrix(y, V, labels=[0, 1]).ravel()
        summ.append({'arm': arm, **{f'{m}_mean': g[m].mean() for m in ('precision', 'recall', 'F1', 'PR-AUC')},
                     **{f'{m}_sd': g[m].std() for m in ('precision', 'recall', 'F1', 'PR-AUC')},
                     'TP_mean': g.TP.mean(), 'FP_mean': g.FP.mean(),
                     'vote_F1': f1_score(y, V), 'vote_precision': tp / max(tp + fp, 1),
                     'vote_recall': tp / (tp + fn), 'vote_TP': tp, 'vote_FP': fp,
                     'meanscore_PR-AUC': average_precision_score(y, S)})
    summ = pd.DataFrame(summ)
    summ.to_csv(f'{OUT}/tables/test_summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(f'\nTEST SET ({len(y)} objects, {y.sum()} TDEs), {R} repetitions')
    print(summ[['arm', 'precision_mean', 'recall_mean', 'F1_mean', 'F1_sd', 'PR-AUC_mean',
                'TP_mean', 'FP_mean', 'vote_F1', 'vote_TP', 'vote_FP', 'meanscore_PR-AUC']]
          .round(3).to_string(index=False))

    # ---- paired SOM effect
    print('\nSOM effect on the test set (With - Without, paired over repetitions)')
    a, b = r[r.arm == 'With SOM'].set_index('repeat'), r[r.arm == 'Without SOM'].set_index('repeat')
    for m in ('F1', 'recall', 'precision', 'PR-AUC'):
        d = a[m] - b.loc[a.index, m]
        print(f'  {m:9s} {d.mean():+.4f}  better in {(d > 0).sum()}/{len(d)}'
              f'  p {stats.ttest_rel(a[m], b.loc[a.index, m]).pvalue:.2g}')

    # ---- figures: per arm, stage 1 (top) + final (bottom), mean +/- sd over repetitions
    for arm, tag in (('Without SOM', 'no_som'), ('With SOM', 'som')):
        g = r[r.arm == arm]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 9.6))
        per = np.array([[[x.s1_agn_rejected, x.s1_agn_passed], [x.s1_tr_rejected, x.s1_tr_passed]]
                        for x in g.itertuples()])
        m = per.mean(0)
        f1 = 2 * m[1, 1] / (2 * m[1, 1] + m[0, 1] + m[1, 0])
        draw_cm(a1, m, per.std(0), ['AGN', 'Transient'], ['AGN', 'Transient'], 'Blues',
                f'Stage 1 (AGN filter)\nF1 = {f1:.3f}')
        per = np.array([[[x.TN, x.FP], [x.FN, x.TP]] for x in g.itertuples()])
        draw_cm(a2, per.mean(0), per.std(0), ['Non-TDE', 'TDE'], ['Non-TDE', 'TDE'], 'Greens',
                f'Final classification\nF1 = {g.F1.mean():.3f} $\\pm$ {g.F1.std():.3f}')
        save(fig, f'{FIGDIR}/test_cm_{tag}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        V = np.mean([calls[arm][k] for k in sorted(calls[arm])], axis=0) >= .5
        cm = confusion_matrix(y, V, labels=[0, 1])
        tp, fp = cm[1, 1], cm[0, 1]
        draw_cm(ax, cm.astype(float), None, ['Non-TDE', 'TDE'], ['Non-TDE', 'TDE'], 'Greens',
                f'{arm}   F1 = {f1_score(y, V):.3f}\nprecision {tp / (tp + fp):.3f}   '
                f'recall {tp / y.sum():.3f}')
    save(fig, f'{FIGDIR}/test_ensemble_cm')
    print(f'\ntables -> {OUT}/tables/')


if __name__ == '__main__':
    main()
