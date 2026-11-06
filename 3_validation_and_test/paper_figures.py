"""
Paper figures for the final configuration (results_wincol_s1 + _s21, 40 repetitions
of 5-fold CV), from the saved out-of-fold predictions - no refitting.

    python paper_figures.py            -> paper_figs/*.pdf and *.png

    fig_stage1_cm        stage 1 (AGN filter), mean +/- sd over repetitions
    fig_final_cm         final TDE vs non-TDE, single model, mean +/- sd
    fig_ensemble_cm      final, majority vote over the 40 repetitions
    fig_ensemble_f1      F1 vs number of repetitions combined

Stage-1 rejection is read from the saved scores (rejected <=> final score 0),
which reproduces the pipeline's own stage-1 counts exactly.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

RUN = 'results_wincol'
OUT = 'paper_figs'
ARMS = ['Without SOM', 'With SOM']
COL = {'With SOM': '#D55E00', 'Without SOM': '#0072B2'}
KS = (1, 2, 3, 5, 10, 20, 40)
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'pdf.fonttype': 42})


def save(fig, name):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT}/{name}.pdf')


def draw_cm(ax, m, sd, labels, xlabels, cmap, title):
    pct = m / m.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            txt = f'{m[i, j]:.0f}' if sd is None else f'{m[i, j]:.1f} $\\pm$ {sd[i, j]:.1f}'
            ax.text(j, i, f'{txt}\n({pct[i, j]:.1f}%)', ha='center', va='center', fontsize=14,
                    color='white' if pct[i, j] > 55 else 'black')
    ax.set_xticks([0, 1], xlabels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    o = pd.concat([pd.read_csv(f'{RUN}_{s}/tables/oof_predictions.csv') for s in ('s1', 's21')],
                  ignore_index=True)
    lab = o.drop_duplicates('object_id').set_index('object_id').SpecType
    V = {a: o[o.arm == a].pivot(index='object_id', columns='repeat', values='pred') for a in ARMS}
    S = {a: o[o.arm == a].pivot(index='object_id', columns='repeat', values='score') for a in ARMS}
    idx = V['With SOM'].index
    t = np.asarray(idx.map(lab)).astype(str)
    y = (t == 'TDE').astype(int)
    agn = t == 'AGN'
    R = V['With SOM'].shape[1]

    # stage 1: rows AGN / transient, columns rejected / passed
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        rej = (S[arm].reindex(idx) == 0).to_numpy()
        per = np.array([[[(agn & rej[:, r]).sum(), (agn & ~rej[:, r]).sum()],
                         [(~agn & rej[:, r]).sum(), (~agn & ~rej[:, r]).sum()]] for r in range(R)])
        m, sd = per.mean(0), per.std(0)
        tp, fn, fp = m[1, 1], m[1, 0], m[0, 1]
        f1 = 2 * tp / (2 * tp + fp + fn)
        draw_cm(ax, m, sd, ['AGN', 'Transient'], ['AGN', 'Transient'], 'Blues',
                f'{arm}   F1 = {f1:.3f}\nAGN rejection {m[0, 0] / m[0].sum():.3f}')
    save(fig, 'fig_stage1_cm')

    # final, single model: mean +/- sd over repetitions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        P = V[arm].reindex(idx).to_numpy().astype(int)
        per = np.array([confusion_matrix(y, P[:, r], labels=[0, 1]) for r in range(R)])
        f1 = np.array([f1_score(y, P[:, r]) for r in range(R)])
        draw_cm(ax, per.mean(0), per.std(0), ['Non-TDE', 'TDE'], ['Non-TDE', 'TDE'], 'Greens',
                f'{arm}   F1 = {f1.mean():.3f} $\\pm$ {f1.std():.3f}')
    save(fig, 'fig_final_cm')

    # final, ensemble: majority vote over all repetitions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        pred = (V[arm].reindex(idx).mean(1) >= .5).astype(int).to_numpy()
        m = confusion_matrix(y, pred, labels=[0, 1])
        draw_cm(ax, m, None, ['Non-TDE', 'TDE'], ['Non-TDE', 'TDE'], 'Greens',
                f'{arm}   F1 = {f1_score(y, pred):.3f}\n'
                f'precision {precision_score(y, pred):.3f}   recall {recall_score(y, pred):.3f}')
    save(fig, 'fig_ensemble_cm')

    # one figure per arm: stage 1 (left) and final (right), mean +/- sd over repetitions
    for arm, tag in (('Without SOM', 'no_som'), ('With SOM', 'som')):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 9.6))
        rej = (S[arm].reindex(idx) == 0).to_numpy()
        per = np.array([[[(agn & rej[:, r]).sum(), (agn & ~rej[:, r]).sum()],
                         [(~agn & rej[:, r]).sum(), (~agn & ~rej[:, r]).sum()]] for r in range(R)])
        m = per.mean(0)
        f1 = 2 * m[1, 1] / (2 * m[1, 1] + m[0, 1] + m[1, 0])
        draw_cm(a1, m, per.std(0), ['AGN', 'Transient'], ['AGN', 'Transient'], 'Blues',
                f'Stage 1 (AGN filter)\nF1 = {f1:.3f}')
        P = V[arm].reindex(idx).to_numpy().astype(int)
        per = np.array([confusion_matrix(y, P[:, r], labels=[0, 1]) for r in range(R)])
        f1 = np.array([f1_score(y, P[:, r]) for r in range(R)])
        draw_cm(a2, per.mean(0), per.std(0), ['Non-TDE', 'TDE'], ['Non-TDE', 'TDE'], 'Greens',
                f'Final classification\nF1 = {f1.mean():.3f} $\\pm$ {f1.std():.3f}')
        save(fig, f'cm_{tag}')

    # F1 vs ensemble size
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ks = [k for k in KS if k <= R]
    end = {}
    for arm in ARMS:
        M = V[arm].reindex(idx).to_numpy()
        mean, lo, hi = [], [], []
        for K in ks:
            f = [f1_score(y, M[:, rng.choice(R, K, replace=False)].mean(1) >= .5)
                 for _ in range(1 if K == R else 500)]
            mean.append(np.mean(f)); lo.append(np.percentile(f, 16)); hi.append(np.percentile(f, 84))
        end[arm] = mean[-1]
        ax.fill_between(ks, lo, hi, color=COL[arm], alpha=.15, lw=0)
        ax.plot(ks, mean, 'o-', color=COL[arm], lw=2, ms=7, label=arm)
    ax.set_xscale('log')
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel('Number of repetitions combined (majority vote)')
    ax.set_ylabel('F1-score')
    ax.set_title(f'With SOM F1 = {end["With SOM"]:.3f},  Without SOM F1 = '
                 f'{end["Without SOM"]:.3f}  ({R} repetitions)', fontsize=13)
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig_ensemble_f1')


if __name__ == '__main__':
    main()
