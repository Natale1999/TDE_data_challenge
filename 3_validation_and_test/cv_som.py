#!/usr/bin/env python
"""
Repeated stratified CV - SOM vs no-SOM.


Design. The 2417 training + 603 validation objects are pooled (3020, 147
TDEs). Outer loop is RepeatedStratifiedKFold: within a repeat the 5 folds
partition the pool, so every object is validated exactly once by a model
that never saw it. Metrics are computed once per repeat over the whole
pool, so each datapoint rests on all 147 TDEs rather than ~29.

Nesting - per outer fold, everything fitted is fitted inside the training
portion:
    StratifiedKFold(5) on the training portion  -> SOM class fractions, OOF
    cross_val_predict(cv=3)                     -> stage-1 5th-percentile cut
    StratifiedKFold(4)                          -> expert probs for the meta
    LogisticRegressionCV(cv=5)                  -> picks C
    cross_val_predict(cv=5)                     -> stage-3 threshold

Six SOMs per outer fold: 5 inner ones for the training rows' features, 1 on
the capped full training portion to score the held-out fold. Every SOM is
fitted on a class-capped subset (150 per class).
"""

# xgboost MUST be imported before QNPy_Latte: torch ships its own libomp and the
# duplicate OpenMP runtimes segfault the process during XGBoost's thread setup.
from xgboost import XGBClassifier

import argparse
import glob
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')          # headless compute nodes: never try to open a window
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import convolve2d

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, precision_recall_curve,
                             average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

from QNPy_Latte.Clustering_with_SOM import (SOM_1D, Load_Light_Curves,
                                            Pad_Light_Curves)


# ----------------------------------------------------------------------------
# 1 - Configuration
# ----------------------------------------------------------------------------


FAST          = False      # False: 3000/700 trees.  True: 500/250, ~2x faster
N_SPLITS      = 5          # outer folds per repeat
N_REPEATS     = 5          # each repeat = one full pass over the pool
SOM_FOLDS     = 5          # inner folds for the out-of-fold SOM features
INCLUDE_P_TDE = True       # keep the raw single-node TDE fraction as a stage-2 feature

SOM_X = SOM_Y = 11         # SOM grid
SOM_EPOCHS    = 250_000
CAP           = 150        # per-class cap before every SOM fit
N_MAX         = 300        # light-curve padding length

N_JOBS        = -1         # n_jobs handed to the estimators

# Epoch window fed to the SOM, in days from peak
SOM_WINDOW    = (0,100)                         # e.g. (0, 100)

# Scalar colour values appended to the SOM input.
SOM_APPEND_COLS  = []                         # e.g. ['cri_p50']
SOM_APPEND_SHARE = 0.30
SOM_APPEND_SHARES = []                        # per-column shares, e.g. [0.30, 0.20]
COLOUR_FILE      = 'colour_features.csv'

FIRST_REPEAT  = 1    # run repeats FIRST_REPEAT .. FIRST_REPEAT+N_REPEATS-1
STAGE1_PCT    = 5    # stage-1 cut: percentile of training-TDE scores kept above
STAGE1_MODEL  = 'rf' # stage-1 transient-vs-AGN model: 'rf' or 'xgb'
# Peak absolute magnitude per band (compute_peak_flux.py: brightest de-extincted
# point -> AB mag -> minus DM(z), plus 2.5 log10(1+z)). Stage 2/3 only; stage 1
# excludes them as it excludes Z, so stage 1 - and what stage 2/3 train on - is
# unchanged. Component test: F1 +0.031 (4/5), SN false positives -7.
PEAK_LUM      = False
DROP_BLOCK    = None   # --drop-block feets|fits|color|z: feature-block ablation
ONLY_ARM      = None   # --only-arm with: run only the With-SOM arm
FINAL         = False  # --final: per seed, train once on ALL training objects, score the test set
TEST          = False  # --test: every fold model also scores the external test set
TEST_DIR      = 'stacked_curves_short_test'
TEST_LOG      = 'test_log.csv'          # Z only - the copy WITHOUT labels
TEST_COLOUR   = 'colour_features_test.csv'
_APPEND_STATS = None   # training mean/sd/scale of the appended colour, reused for test
SECOND_EXPERT = 'rf'   # second stage-2 expert: 'rf' or 'brf' (balanced random forest)
THIRD_EXPERT  = 'et'   # third stage-2 expert: 'et' (ExtraTrees) or 'lgbm' (LightGBM)
META_S1       = False  # pass stage 1's transient score to the meta-learner
FIX_TREES     = False  # RF/ET: balance once (sample weights only), min_samples_leaf=3
META_INPUTS   = 'all'  # meta-learner inputs: 'all' = 12 expert probabilities, 'ptde' = 3 P(TDE)
META_SCORING  = None   # meta-learner C chosen by this CV score; None = log-loss
SAVE_META     = False  # save each fold's meta-learner inputs to tables/meta/ (meta_lab.py)
DUMP_TABLES   = False  # save each fold's tables + stage-1 masks to tables/dump/ (stack_lab.py)
PEAK_FILE     = 'peak_flux_train.csv'
PEAK_COLS     = [f'M_peak_{b}' for b in 'ugrizy']
SOM_TAU_GRID  = np.linspace(-150, 200, 100)   # the grid save_stacked_curves wrote
SOM_BANDS     = 3                             # g, r, i concatenated in that order

TARGET_CLASSES = ['SN Ia', 'TDE', 'SN II', 'AGN']
TRAIN_DIR = 'stacked_curves_short_all_train'
VAL_DIR   = 'stacked_curves_short_all_val'

C_WO, C_W = '#0072B2', '#D55E00'     # colourblind-safe pair for the figures

# Output layout, filled in by main().
OUT = FIG_DIR = TAB_DIR = None

# Populated by load_inputs() / pool_light_curves().
df_features = None
spec_map    = None
y_tde_pool  = None

BANDS = 'rgizuy'


# One line per event, appended by every task to a single shared file, so a
# 20-way array run reads as one timeline instead of 20 logs. Short writes to a
# file opened O_APPEND are atomic on POSIX, so tasks interleave lines but never
# corrupt each other. TAG identifies which repeat wrote each line.
PROGRESS = None
TAG = ''
VERBOSE = True      # False in shard mode: 20 tasks sharing one log should not
                    # each reprint the same banner, config and column lists


def log(msg=''):
    print(msg, flush=True)


def vlog(msg=''):
    """Startup chatter: useful once, noise repeated 20 times."""
    if VERBOSE:
        print(msg, flush=True)


def note(msg):
    """A progress line: to stdout, and to the shared log if one is configured."""
    line = f'{time.strftime("%H:%M:%S")} {TAG}{msg}'
    print(line, flush=True)
    if PROGRESS:
        try:
            with open(PROGRESS, 'a') as fh:
                fh.write(line + '\n')
        except OSError:
            pass        # a progress line is never worth killing a run over


# ----------------------------------------------------------------------------
# 2 - Preflight: does the data this run needs actually exist?
# ----------------------------------------------------------------------------
def preflight():
    """Check every input up front. A 25-fold run is too long to die an hour in."""
    missing, notes = [], []

    for f in ['train_log.csv', 'df_features_train.csv', 'df_features_val.csv']:
        if not os.path.isfile(f):
            missing.append(f)

    for b in BANDS:
        f = f'feature_df_{b}.csv'
        if not os.path.isfile(f):
            missing.append(f)

    for d in [TRAIN_DIR, VAL_DIR]:
        if not os.path.isdir(d):
            missing.append(d + '/')
        else:
            n = len(glob.glob(os.path.join(d, '*.csv')))
            notes.append(f'  {d}/  {n} light curves')
            if n == 0:
                missing.append(d + '/*.csv')

    if notes:
        vlog('inputs found:')
        for n in notes:
            vlog(n)

    if missing:
        log('')
        log('!' * 76)
        log('MISSING INPUTS - cannot run:')
        for m in missing:
            log(f'  {m}')
        log('')
        log('The six feature_df_<band>.csv files hold the per-band photometric')
        log('features (Mean, Std, StetsonK, ... per band). Section 3 merges them')
        log('into df_features, and add_derived() builds every colour ratio from')
        log('their Mean_*/Std_*/StetsonK_g columns, so the run cannot start')
        log('without them. They are not derivable from the data in this folder:')
        log('the stacked curves here carry only g, r and i, while these files')
        log('cover r, g, i, z, u and y. Copy them from the original project.')
        log('!' * 76)
        sys.exit(2)


# ----------------------------------------------------------------------------
# 3 - Labels and tabular features
# ----------------------------------------------------------------------------
def _band_features(suffix=''):
    """
    Building the feature dataframe
    """
    parts = []
    for f in BANDS:                         
        d = pd.read_csv(f'feature_df_{f}{suffix}.csv').drop(columns=['Unnamed: 0'],
                                                            errors='ignore')
        parts.append(d.rename(columns={c: f'{c}_{f}' for c in d.columns
                                       if c != 'object_id'}))
    out = parts[0]
    for d in parts[1:]:
        out = out.merge(d, on='object_id', how='outer')
    return out


def add_derived(df):
    """
    Computing additional features from the existing ones
    """
    e = 1e-9
    df = df.copy()
    df['ratio_ug'] = df['Mean_u'] / (df['Mean_g'] + e)
    df['ratio_gr'] = df['Mean_g'] / (df['Mean_r'] + e)
    df['ratio_ri'] = df['Mean_r'] / (df['Mean_i'] + e)
    df['ratio_iz'] = df['Mean_i'] / (df['Mean_z'] + e)
    df['ratio_uz'] = df['Mean_u'] / (df['Mean_z'] + e)
    df['std_ratio_ug'] = df['Std_u'] / (df['Std_g'] + e)
    df['std_ratio_gr'] = df['Std_g'] / (df['Std_r'] + e)
    df['stetson_ratio'] = df['StetsonK_g'] / (df['Mean_g'] + e)
    df['color_stability_ug'] = (df['Std_u'] / (df['Mean_u'] + e)) / (df['Std_g'] / (df['Mean_g'] + e) + e)
    df['color_stability_gr'] = (df['Std_g'] / (df['Mean_g'] + e)) / (df['Std_r'] / (df['Mean_r'] + e) + e)
    return df.replace([np.inf, -np.inf], np.nan)


def load_inputs():
    """Loading labels + the merged tabular feature table."""
    global df_features, spec_map

    train_log = pd.read_csv('train_log.csv')
    spec_map  = train_log.set_index('object_id')['SpecType']

    df_features = _band_features()
    additional  = pd.concat([pd.read_csv('df_features_train.csv'),
                             pd.read_csv('df_features_val.csv')], ignore_index=True)
    df_features = (df_features.merge(additional, on='object_id', how='outer')
                              .drop(columns='Unnamed: 0', errors='ignore')
                              .merge(train_log[['Z', 'object_id']], on='object_id',
                                     how='outer'))
    if PEAK_LUM:
        pk = pd.read_csv(PEAK_FILE, usecols=['object_id'] + PEAK_COLS)
        df_features = df_features.merge(pk, on='object_id', how='left')
    vlog(f'df_features {df_features.shape}')
    if TEST or FINAL:
        load_test_features()


def load_test_features():
    """The test set's tabular features, built exactly like df_features. No labels:
    Z comes from TEST_LOG, which has SpecType empty. Missing values in
    df_features_test.csv are -9999 where the training file has NaN; every model
    input is fillna(-9999), so both reach the models identically."""
    global df_features_test
    t_log = pd.read_csv(TEST_LOG)
    d = _band_features('_test')
    d = (d.merge(pd.read_csv('df_features_test.csv'), on='object_id', how='outer')
          .drop(columns='Unnamed: 0', errors='ignore')
          .merge(t_log[['Z', 'object_id']], on='object_id', how='outer'))
    df_features_test = d.replace(-9999, np.nan)[df_features.columns]
    vlog(f'df_features_test {df_features_test.shape}')


# ----------------------------------------------------------------------------
# 4 - SOM machinery
# ----------------------------------------------------------------------------
def get_gaussian_kernel(size=3, sigma=1.0):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    g  = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    k  = np.outer(g, g)
    return k / k.sum()


KERNEL_GAUSS = get_gaussian_kernel(3, 1.0)
KERNEL_BOX   = np.ones((3, 3)) / 9.0


def coarse_label(s):
    """4-class grouping: the classifier target, and the fold stratification."""
    if 'TDE' in s:   return 'TDE'
    if 'AGN' in s:   return 'AGN'
    if 'SN Ia' in s: return 'SN Ia'
    return 'SN II'


def _smooth(frac, k):
    """NaN-aware normalised convolution: empty nodes neither contribute nor absorb."""
    m   = np.isfinite(frac).astype(float)
    num = convolve2d(np.nan_to_num(frac), k, mode='same', boundary='fill', fillvalue=0)
    den = convolve2d(m,                   k, mode='same', boundary='fill', fillvalue=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = num / den
    s[den == 0] = np.nan
    return s


def _node_maps(bi, bj, labels):
    """Per-node class fractions: raw, 3x3 box mean, gaussian kernel."""
    labels = np.asarray(labels)
    n_all  = np.zeros((SOM_X, SOM_Y))
    np.add.at(n_all, (bi, bj), 1.0)
    maps = {}
    for cls in TARGET_CLASSES:
        n_c = np.zeros((SOM_X, SOM_Y))
        np.add.at(n_c, (bi, bj), (labels == cls).astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            raw = np.where(n_all > 0, n_c / np.maximum(n_all, 1), np.nan)
        maps[cls] = {'raw': raw,
                     'box': _smooth(raw, KERNEL_BOX),
                     'kernel': _smooth(raw, KERNEL_GAUSS)}
    return maps, n_all


def fit_som_maps(mags, labels, seed, epochs=None):
    epochs = SOM_EPOCHS if epochs is None else epochs
    som, *_ = SOM_1D(mags, som_x=SOM_X, som_y=SOM_Y, pca_init=False, save_som=False,
                     train_mode='random', epochs=epochs, learning_rate=0.1, sigma=1.0,
                     batch_size=300, random_seed=seed,
                     plot_frequency=epochs + 10)     # skips the costly error eval
    b = np.array([som.winner(c) for c in mags])
    maps, occ = _node_maps(b[:, 0], b[:, 1], labels)
    return som, maps, occ


def apply_som_maps(som, maps, mags, ids):
    W = som.get_weights()
    b = np.array([som.winner(c) for c in mags])
    out = pd.DataFrame({
        'object_id':       np.asarray(ids),
        'BMU_i':           b[:, 0],
        'BMU_j':           b[:, 1],
        'Distance_to_BMU': [np.linalg.norm(c - W[tuple(k)]) for c, k in zip(mags, b)],
    })
    out['P_TDE']          = maps['TDE']['raw'][b[:, 0], b[:, 1]]
    out['P_TDE_weighted'] = maps['TDE']['box'][b[:, 0], b[:, 1]]
    for cls in ['TDE', 'SN Ia', 'SN II', 'AGN']:
        out[f'P_{cls}_weighted_kernel'] = maps[cls]['kernel'][b[:, 0], b[:, 1]]
    return out


def cap_per_class(labels, cap=None, seed=42):
    """Capping every class at max 150 objects, for the SOM fit. Returns the indices of the kept objects."""
    cap    = CAP if cap is None else cap
    rng    = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx    = np.arange(len(labels))
    keep   = [rng.permutation(idx[labels == c])[:cap] if (labels == c).sum() > cap
              else idx[labels == c] for c in pd.unique(labels)]
    return np.sort(np.concatenate(keep))


def oof_som_features(mags, ids, labels, seed, n_splits=None, cap=None):
    """
    Out-of-fold SOM features. Each inner fold caps its 4/5 by class, fits a SOM
    and its maps on that balanced subset, then scores the held-out 1/5.
    No object contributes to the class fractions used as its own feature.
    """
    n_splits = SOM_FOLDS if n_splits is None else n_splits
    cap      = CAP if cap is None else cap
    ids, labels = np.asarray(ids), np.asarray(labels)
    strata = np.array([coarse_label(s) for s in labels])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(ids)), strata), 1):
        t0 = time.time()
        sub = tr[cap_per_class(labels[tr], cap=cap, seed=seed + k)]
        som, maps, _ = fit_som_maps([mags[i] for i in sub], labels[sub], seed=seed)
        parts.append(apply_som_maps(som, maps, [mags[i] for i in te], ids[te]))
        note(f'  som {k}/{n_splits + 1}  {len(sub):4d} curves  {time.time()-t0:5.0f}s')
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# 5 - Light curves
# ----------------------------------------------------------------------------
def load_curves(folder, n_max=None):
    n_max = N_MAX if n_max is None else n_max
    lc, ids = Load_Light_Curves(folder, one_filter=True)
    if len(lc) == 0:
        raise RuntimeError(
            f'Load_Light_Curves found nothing in {folder!r}. If QNPy_Latte was '
            'reinstalled, its Windows-only glob separator is back - re-apply '
            'Linux_Env_Setup/patch_qnpy.py.')
    pad  = Pad_Light_Curves(lc, n_max)
    mags = [np.nan_to_num(d['mag'].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            for d in pad]
    # Load_Light_Curves keeps the separator: '/df_NAME' on unix, '\\df_NAME' on Windows
    ids  = pd.Series(ids).str.replace(r'^[\\/]*df_', '', regex=True).to_numpy()
    return mags, ids


def pool_light_curves(cache=None):
    """Section 5: pool train + val light curves and attach labels."""
    global y_tde_pool

    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        pool_mags, pool_ids = list(z['mags']), z['ids']
        vlog(f'light curves from cache {cache}  ({len(pool_ids)} objects)')
    else:
        pool_mags, pool_ids = [], []
        for folder in [TRAIN_DIR, VAL_DIR]:
            m, i = load_curves(folder)
            pool_mags += m
            pool_ids  += list(i)
        pool_ids = np.array(pool_ids)
        if cache:
            # Written via a unique temp file then renamed: array tasks start
            # together and would otherwise interleave writes into one half-
            # written npz. 
            tmp = f'{cache}.{os.getpid()}.tmp.npz'
            np.savez_compressed(tmp, mags=np.array(pool_mags), ids=pool_ids)
            os.replace(tmp, cache)
            vlog(f'light curves cached to {cache}')

    if SOM_WINDOW is not None:
        lo, hi = SOM_WINDOW
        m = (SOM_TAU_GRID >= lo) & (SOM_TAU_GRID <= hi)
        n = len(SOM_TAU_GRID)
        # each vector is g(100) | r(100) | i(100); keep the same epochs in each
        pool_mags = [v.reshape(SOM_BANDS, n)[:, m].ravel() for v in pool_mags]
        vlog(f'SOM window {lo}..{hi}d -> {m.sum()} of {n} epochs per band, '
             f'{len(pool_mags[0])} dims')

    if SOM_APPEND_COLS:
        # Append a scalar color values to the SOM input. 
        cf = pd.read_csv(COLOUR_FILE).set_index('object_id').reindex(pool_ids)
        C = cf[SOM_APPEND_COLS].to_numpy(float)
        mu, sd = np.nanmean(C, 0), np.nanstd(C, 0)
        C = np.nan_to_num((C - mu) / (sd + 1e-9), nan=0.0)
        v_flux = np.asarray(pool_mags).var(0).sum()
        if SOM_APPEND_SHARES:
            # one share per column: var_j = share_j / (1 - sum shares) * V_flux,
            # so adding a colour does not dilute the ones already there
            tot = sum(SOM_APPEND_SHARES)
            s = np.sqrt(np.asarray(SOM_APPEND_SHARES) / (1 - tot) * v_flux)
            desc = ', '.join(f'{c} {sh:.0%}' for c, sh in
                             zip(SOM_APPEND_COLS, SOM_APPEND_SHARES))
        else:
            # one share for the whole block, split evenly across its columns
            s = np.sqrt(SOM_APPEND_SHARE / (1 - SOM_APPEND_SHARE) * v_flux / C.shape[1])
            desc = f'{SOM_APPEND_COLS} at {SOM_APPEND_SHARE:.0%}'
        pool_mags = [np.concatenate([v, s * c]) for v, c in zip(pool_mags, C)]
        vlog(f'SOM input + {desc} of distance variance -> {len(pool_mags[0])} dims')
        global _APPEND_STATS
        _APPEND_STATS = (mu, sd, s)

    y_pool = pd.Series(pool_ids).map(spec_map).to_numpy()
    keep   = pd.notna(y_pool)
    pool_mags = [m for m, k in zip(pool_mags, keep) if k]
    pool_ids, y_pool = pool_ids[keep], y_pool[keep]

    pool_strata = np.array([coarse_label(s) for s in y_pool])
    y_tde_pool  = pd.Series((y_pool == 'TDE').astype(int), index=pool_ids)
    vlog(f'pooled {len(pool_ids)} objects, {y_tde_pool.sum()} TDEs')
    return pool_mags, pool_ids, y_pool, pool_strata


def test_light_curves():
    """Test grids -> SOM input vectors, transformed exactly like the pool: same
    window, and the colour term standardised and scaled with the TRAINING
    statistics, so a test object lands where an identical training object would."""
    mags, ids = load_curves(TEST_DIR)
    if SOM_WINDOW is not None:
        lo, hi = SOM_WINDOW
        m, n = (SOM_TAU_GRID >= lo) & (SOM_TAU_GRID <= hi), len(SOM_TAU_GRID)
        mags = [v.reshape(SOM_BANDS, n)[:, m].ravel() for v in mags]
    if SOM_APPEND_COLS:
        mu, sd, s = _APPEND_STATS
        cf = pd.read_csv(TEST_COLOUR).set_index('object_id').reindex(ids)
        C = np.nan_to_num((cf[SOM_APPEND_COLS].to_numpy(float) - mu) / (sd + 1e-9), nan=0.0)
        mags = [np.concatenate([v, s * c]) for v, c in zip(mags, C)]
    vlog(f'test light curves: {len(ids)} objects, {len(mags[0])} dims')
    return mags, np.asarray(ids)


def build_test_table(som_feats):
    """Like build_table, from the test features; no SpecType column at all."""
    d = add_derived(df_features_test.merge(som_feats, on='object_id'))
    return d.sort_values('object_id').reset_index(drop=True)


def build_table(som_feats):
    d = add_derived(df_features.merge(som_feats, on='object_id'))
    d['SpecType'] = d['object_id'].map(spec_map)
    return d[d.SpecType.notna()].sort_values('object_id').reset_index(drop=True)


# ----------------------------------------------------------------------------
# 6 - Feature configuration and the two pipeline stages
# ----------------------------------------------------------------------------
# BMU_i / BMU_j are never model inputs: with a different SOM per fold, node
# (3,4) is a different region of light-curve space each time. Distance_to_BMU
# is SOM-derived, so it is gated with the rest - otherwise the no-SOM baseline
# silently keeps a SOM feature.

som_meta_probs = [
    'P_TDE', 'P_TDE_weighted', 'P_TDE_weighted_kernel',
    'P_SN Ia_weighted_kernel', 'P_SN II_weighted_kernel', 'P_AGN_weighted_kernel'
]
som_derived       = ['Distance_to_BMU']
context_cols      = ['Z', 'Distance_to_BMU', 'BMU_i', 'BMU_j']
always_drop_s1    = ['SpecType', 'object_id'] + context_cols
pass_through_cols = ['Z', 'P_AGN_weighted_kernel']

configs = {}


def build_configs():
    global configs
    base_drop_s2 = ['object_id', 'SpecType', 'BMU_i', 'BMU_j']
    if not INCLUDE_P_TDE:
        base_drop_s2 = base_drop_s2 + ['P_TDE']

    s1_extra = PEAK_COLS if PEAK_LUM else []
    configs = {
        "Without SOM": {"s1_drop": always_drop_s1 + s1_extra + som_meta_probs,
                        "s2_drop": base_drop_s2   + som_meta_probs + som_derived},
        "With SOM":    {"s1_drop": always_drop_s1 + s1_extra,
                        "s2_drop": base_drop_s2},
    }
    if DROP_BLOCK:
        # --drop-block: feature-block ablation, the block is removed from every stage
        # (for z this also removes it from the meta-learner pass-through)
        cols = ablation_block(DROP_BLOCK)
        for cfg in configs.values():
            cfg['s1_drop'] = cfg['s1_drop'] + cols
            cfg['s2_drop'] = cfg['s2_drop'] + cols
        vlog(f'ablation: dropping {DROP_BLOCK} ({len(cols)} columns)')
    if ONLY_ARM == 'with':
        configs = {'With SOM': configs['With SOM']}
    for name, cfg in configs.items():
        vlog(f"{name:11s} s2_drop: {cfg['s2_drop'][:12]}{' ...' if len(cfg['s2_drop']) > 12 else ''}")


def ablation_block(block):
    """Columns of one feature block (the groups of the paper's feature table)."""
    if block == 'feets':
        base = [c for c in pd.read_csv('feature_df_r.csv', nrows=1).columns
                if c not in ('object_id', 'Unnamed: 0')]
        return [f'{c}_{b}' for c in base for b in BANDS]
    if block == 'fits':
        return [c for c in pd.read_csv('df_features_train.csv', nrows=1).columns
                if c not in ('object_id', 'Unnamed: 0', 'Z')]
    if block == 'color':
        return ['ratio_ug', 'ratio_gr', 'ratio_ri', 'ratio_iz', 'ratio_uz', 'std_ratio_ug',
                'std_ratio_gr', 'stetson_ratio', 'color_stability_ug', 'color_stability_gr']
    if block == 'z':
        return ['Z']
    raise ValueError(block)


def get_class_label(spec_type):
    if 'TDE'   in spec_type: return 3
    if 'AGN'   in spec_type: return 0
    if 'SN Ia' in spec_type: return 1
    if 'SN II' in spec_type: return 2
    return 2


def run_stage1(train_df, val_df, drop_cols, seed=42, test_df=None):

    n_est = 500 if FAST else 3000

    X_tr = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).fillna(-9999)
    X_va = val_df.drop(  columns=[c for c in drop_cols if c in val_df.columns]).fillna(-9999)
    X_va = X_va[X_tr.columns]

    y_tr     = train_df.SpecType.astype(str).str.contains('TDE|SN').astype(int)
    y_va     = val_df.SpecType.astype(str).str.contains('TDE|SN').astype(int)
    y_tr_tde = train_df.SpecType.astype(str).str.contains('TDE').astype(int)
    y_va_tde = val_df.SpecType.astype(str).str.contains('TDE').astype(int)

    if STAGE1_MODEL == 'xgb':
        # same settings as stage1_cuts.py: 147 AGN passed vs RF's 258 at 95% of TDEs
        rf = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.7, tree_method='hist',
                           scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                           eval_metric='logloss', random_state=seed, n_jobs=N_JOBS)
    else:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=20, min_samples_leaf=2,
                                    class_weight='balanced_subsample',
                                    random_state=seed, n_jobs=N_JOBS)
    rf.fit(X_tr, y_tr)

    p_tr = cross_val_predict(rf, X_tr, y_tr, cv=3, method='predict_proba',
                             n_jobs=N_JOBS)[:, 1]
    # keep training TDEs down to this percentile of their own OOF scores;
    # lower = looser filter, fewer TDEs lost before stage 2/3, more AGN through
    thr  = np.percentile(p_tr[y_tr_tde == 1], STAGE1_PCT)
    p_va = rf.predict_proba(X_va)[:, 1]

    cm_va = confusion_matrix(y_va, (p_va >= thr).astype(int))
    tn, fp, fn, tp = cm_va.ravel()

    out_te = {}
    if test_df is not None:
        X_te = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).fillna(-9999)
        p_te = rf.predict_proba(X_te[X_tr.columns])[:, 1]
        out_te = {'p_te': p_te, 'test_mask': p_te >= thr}

    return {**out_te, 'model': rf, 'columns': X_tr.columns, 'features': X_tr.shape[1],
            'threshold': thr, 'transient_recall': tp / (tp + fn),
            'agn_rejection': tn / (tn + fp),
            'tde_lost': int(((y_va_tde == 1) & (p_va < thr)).sum()),
            'tde_kept': int(((y_va_tde == 1) & (p_va >= thr)).sum()),
            'f1_train': f1_score(y_tr, (p_tr >= thr).astype(int)),
            'f1_val':   f1_score(y_va, (p_va >= thr).astype(int)),
            'train_mask': p_tr >= thr, 'val_mask': p_va >= thr,
            'p_tr': p_tr, 'p_va': p_va,    # OOF training / test scores
            'cm_train': confusion_matrix(y_tr, (p_tr >= thr).astype(int)),
            'cm_val': cm_va}


def run_stage23(train_df, val_df, train_mask, val_mask, drop_cols, seed=42,
                s1_scores=None, test_df=None, test_mask=None):

    n_est = 250 if FAST else 700

    X_tr = (train_df.loc[train_mask]
            .drop(columns=[c for c in drop_cols if c in train_df.columns])
            .fillna(-9999).reset_index(drop=True))
    X_va = (val_df.loc[val_mask]
            .drop(columns=[c for c in drop_cols if c in val_df.columns]).fillna(-9999))
    X_va = X_va[X_tr.columns]

    y_tr  = train_df.loc[train_mask, 'SpecType'].astype(str).apply(get_class_label).reset_index(drop=True)
    y_bin = (y_tr == 3).astype(int).values

    pcols = [c for c in pass_through_cols
             if c in train_df.columns and c not in drop_cols]
    P_tr = train_df.loc[train_mask, pcols].fillna(-9999).values
    P_va = val_df.loc[val_mask,   pcols].fillna(-9999).values
    if s1_scores is not None:
        # --meta-s1-score: stage 1's transient score as one more meta input -
        # out-of-fold for the training rows, from the training-fold model for
        # the test rows, so no fold ever scores itself
        P_tr = np.column_stack([P_tr, np.asarray(s1_scores[0])[np.asarray(train_mask)]])
        P_va = np.column_stack([P_va, np.asarray(s1_scores[1])[np.asarray(val_mask)]])
        pcols = pcols + ['S1_score']

    # All experts are fitted with balanced sample weights below. class_weight
    # would multiply with them (classes weighted twice), so --fix-trees drops it
    # and uses leaves of >= 3 objects for smoother probabilities.
    tree_kw = ({'min_samples_leaf': 3} if FIX_TREES else {'class_weight': 'balanced'})
    experts = [
        XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.02, subsample=0.8,
                      colsample_bytree=0.7, objective='multi:softprob', num_class=4,
                      tree_method='hist', n_jobs=N_JOBS, random_state=seed),
        RandomForestClassifier(n_estimators=n_est, max_depth=20, criterion='log_loss',
                               **tree_kw, n_jobs=N_JOBS, random_state=seed),
        ExtraTreesClassifier(n_estimators=n_est, max_depth=20,
                             **tree_kw, n_jobs=N_JOBS, random_state=seed)]
    if THIRD_EXPERT == 'lgbm':
        # --third-expert lgbm: LightGBM in ET's slot, set up like the XGB expert
        # (700 trees, lr 0.02, row/column subsampling 0.8/0.7, balanced sample weights)
        from lightgbm import LGBMClassifier
        experts[2] = LGBMClassifier(n_estimators=n_est, learning_rate=0.02, num_leaves=31,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                    objective='multiclass', n_jobs=N_JOBS,
                                    random_state=seed, verbose=-1)
    # --second-expert brf / --third-expert brf: balanced random forest in RF's or
    # ET's slot. Every tree is grown on its own class-balanced undersample, so it is
    # fitted WITHOUT the balanced sample weights the other experts get (that would
    # balance twice)
    def _brf():
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(
            n_estimators=n_est, max_depth=20, sampling_strategy='all', replacement=True,
            bootstrap=False, n_jobs=N_JOBS, random_state=seed)
    if SECOND_EXPERT == 'brf':
        experts[1] = _brf()
    if THIRD_EXPERT == 'brf':
        experts[2] = _brf()
    enames = ('XGB', 'BRF' if SECOND_EXPERT == 'brf' else 'RF',
              {'lgbm': 'LGBM', 'brf': 'BRF'}.get(THIRD_EXPERT, 'ET'))

    def _fit(m, X, y, w):
        if type(m).__name__ == 'BalancedRandomForestClassifier':
            return clone(m).fit(X, y)
        return clone(m).fit(X, y, sample_weight=w)

    oof = np.zeros((len(X_tr), 12))
    for tr_i, va_i in StratifiedKFold(4, shuffle=True, random_state=seed).split(X_tr, y_tr):
        w = compute_sample_weight('balanced', y_tr.iloc[tr_i])
        for k, m in enumerate(experts):
            f = _fit(m, X_tr.iloc[tr_i], y_tr.iloc[tr_i], w)
            oof[va_i, 4*k:4*k+4] = f.predict_proba(X_tr.iloc[va_i])

    w_full = compute_sample_weight('balanced', y_tr)
    fitted = [_fit(m, X_tr, y_tr, w_full) for m in experts]

    # expert columns: 4 class probabilities per expert (AGN, SN Ia, SN II, TDE);
    # --meta-inputs ptde keeps only each expert's P(TDE)
    E_va = np.hstack([f.predict_proba(X_va) for f in fitted])
    cols = [3, 7, 11] if META_INPUTS == 'ptde' else list(range(12))
    meta_X_tr = np.hstack([oof[:, cols], P_tr])
    meta_X_va = np.hstack([E_va[:, cols], P_va])

    meta = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', penalty='l2',
                                max_iter=3500, n_jobs=N_JOBS, random_state=seed,
                                scoring=META_SCORING)
    meta.fit(meta_X_tr, y_bin)

    # threshold from out-of-fold meta predictions, never in-sample
    oof_meta  = cross_val_predict(meta, meta_X_tr, y_bin, cv=5, method='predict_proba')[:, 1]
    val_score = meta.predict_proba(meta_X_va)[:, 1]

    pr, rc, th = precision_recall_curve(y_bin, oof_meta)
    thr = float(th[np.argmax(2 * pr * rc / (pr + rc + 1e-9))])

    y_va_full = val_df.SpecType.astype(str).str.contains('TDE').astype(int).values
    score = np.zeros(len(val_df));      score[np.where(val_mask)[0]] = val_score
    pred  = np.zeros(len(val_df), int); pred[np.where(val_mask)[0]]  = (val_score >= thr).astype(int)
    cm_va = confusion_matrix(y_va_full, pred)
    tn, fp, fn, tp = cm_va.ravel()

    # --test: the same fitted experts, meta-learner and thresholds on the test set
    test_out = None
    if test_df is not None:
        if s1_scores is not None:
            raise NotImplementedError('--test with --meta-s1-score')
        tm = np.asarray(test_mask)
        X_te = (test_df.loc[tm].drop(columns=[c for c in drop_cols if c in test_df.columns])
                .fillna(-9999))[X_tr.columns]
        P_te = test_df.loc[tm, pcols].fillna(-9999).values
        E_te = np.hstack([f.predict_proba(X_te) for f in fitted])
        s_te = meta.predict_proba(np.hstack([E_te[:, cols], P_te]))[:, 1]
        sc = np.zeros(len(test_df)); sc[tm] = s_te
        test_out = (sc, (sc >= thr).astype(int) * tm)

    # meta-learner weights; expert columns follow predict_proba's label order
    # (0 AGN, 1 SN Ia, 2 SN II, 3 TDE). scoef = coef * input sd, comparable across inputs
    names = [f'{e}_{c}' for e in enames for c in ('AGN', 'SNIa', 'SNII', 'TDE')]
    names = [names[i] for i in cols] + pcols
    coef = {f'coef_{n}': float(v) for n, v in zip(names, meta.coef_.ravel())}
    coef.update({f'scoef_{n}': float(v) for n, v in
                 zip(names, meta.coef_.ravel() * meta_X_tr.std(0))})
    coef['coef_intercept'], coef['meta_C'] = float(meta.intercept_[0]), float(meta.C_[0])

    return {'features': X_tr.shape[1], 'threshold': round(thr, 4), 'coef': coef,
            'test': test_out,
            # everything the meta-learner saw, so it can be refitted offline:
            # training rows = out-of-fold expert probabilities (+ pass-through),
            # test rows = full-fit expert probabilities on the held-out fold
            'meta_in': {'X_tr': meta_X_tr, 'y_tr': y_bin, 'X_va': meta_X_va,
                        'y_va_full': y_va_full, 'val_mask': np.asarray(val_mask),
                        'names': names},
            'tde_recall':    round(tp / (tp + fn), 4),
            'tde_precision': round(tp / (tp + fp + 1e-9), 4),
            'f1_train': f1_score(y_bin, (oof_meta >= thr).astype(int)),
            'f1_val':   f1_score(y_va_full, pred),
            'pr_auc':   average_precision_score(y_va_full, score),
            'tde_lost': int(fn), 'fp': int(fp),
            'cm_train': confusion_matrix(y_bin, (oof_meta >= thr).astype(int)),
            'cm_val': cm_va,
            'val_scores': score, 'val_preds': pred}


# ----------------------------------------------------------------------------
# 7 - Reporting
# ----------------------------------------------------------------------------
def pool_repeats(store, done):
    """Metrics per completed repeat, over every object in the pool."""
    rows = []
    for arm, reps in store.items():
        for rep, d in reps.items():
            if rep not in done:
                continue
            oid    = list(d)
            score  = np.array([d[o][0] for o in oid])
            pred   = np.array([d[o][1] for o in oid]).astype(int)
            y_true = y_tde_pool.loc[oid].values
            rows.append({'repeat': rep, 'arm': arm,
                         'n': len(oid), 'n_TDE': int(y_true.sum()),
                         'precision': precision_score(y_true, pred, zero_division=0),
                         'recall':    recall_score(y_true, pred),
                         'F1':        f1_score(y_true, pred),
                         'PR-AUC':    average_precision_score(y_true, score)})
    return pd.DataFrame(rows).sort_values(['arm', 'repeat'])


def report(reps_df, tag='latest'):
    log('\n' + '=' * 76)
    log('PER REPEAT  (pooled over all folds - every object validated once)')
    log(reps_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    log(f'\nMEAN +/- SD over {reps_df.repeat.nunique()} repeat(s)')
    log(reps_df.groupby('arm')[['precision', 'recall', 'F1', 'PR-AUC']]
               .agg(['mean', 'std']).to_string(float_format=lambda v: f'{v:.4f}'))

    piv = reps_df.pivot_table(index='repeat', columns='arm',
                              values=['precision', 'recall', 'F1', 'PR-AUC'])
    n = len(piv)
    log('\nPAIRED DELTA  (With SOM - Without SOM, same repeat)')
    for m in ['precision', 'recall', 'F1', 'PR-AUC']:
        d = piv[m]['With SOM'] - piv[m]['Without SOM']
        if n > 1:
            se = d.std(ddof=1) / np.sqrt(n)
            log(f'  {m:10s} mean {d.mean():+.4f}  sd {d.std():.4f}  '
                f'CI [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]  '
                f'wins {int((d>0).sum())}/{n}')
        else:
            log(f'  {m:10s} {d.iloc[0]:+.4f}   (n=1, no CI)')

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for col, m in enumerate(['F1', 'PR-AUC']):
        p = reps_df.pivot_table(index='repeat', columns='arm', values=m)
        ax[0, col].plot(p.index, p['Without SOM'], 'o-', color=C_WO, lw=2, ms=8,
                        label='Without SOM')
        ax[0, col].plot(p.index, p['With SOM'], 'o-', color=C_W, lw=2, ms=8,
                        label='With SOM')
        ax[0, col].set_xlabel('repeat'); ax[0, col].set_ylabel(m)
        ax[0, col].set_title(f'Pooled {m} per repeat ({int(reps_df.n_TDE.iloc[0])} TDEs)')
        ax[0, col].set_xticks(p.index); ax[0, col].grid(alpha=.3)
        ax[0, col].tick_params(axis='x', labelrotation=90)
        if col == 0:
            ax[0, col].legend(frameon=False)

        d = p['With SOM'] - p['Without SOM']
        a = ax[1, col]
        a.axhline(0, color='#888', lw=1)
        if n > 1:
            se = d.std(ddof=1) / np.sqrt(n)
            a.axhspan(d.mean()-1.96*se, d.mean()+1.96*se, color=C_W, alpha=.15)
            a.axhline(d.mean(), color=C_W, lw=2)
        a.bar(d.index, d.values,
              color=[C_W if v > 0 else C_WO for v in d.values], alpha=.85)
        a.set_xlabel('repeat'); a.set_ylabel(f'D{m}')
        a.set_title(f'delta {m}  mean {d.mean():+.4f}' +
                    (f'  wins {int((d>0).sum())}/{n}' if n > 1 else '  (n=1)'))
        a.set_xticks(d.index); a.grid(alpha=.3, axis='y')
        a.tick_params(axis='x', labelrotation=90)

    fig.suptitle(f'Repeated stratified {N_SPLITS}-fold CV - {n} repeat(s), '
                 f'FAST={FAST}, SOM {SOM_X}x{SOM_Y}', fontsize=14)
    fig.tight_layout()
    # The notebook overwrote one figure per repeat. In a batch run nobody is
    # watching, so keep a per-repeat snapshot as well as the rolling 'latest'.
    fig.savefig(os.path.join(FIG_DIR, 'cv_summary.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'cv_summary_{tag}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def confusion_figure(oof_store):
    cms = {}
    for arm, reps in oof_store.items():
        mats = []
        for rep, d in reps.items():
            oid    = list(d)
            pred   = np.array([d[o][1] for o in oid]).astype(int)
            y_true = y_tde_pool.loc[oid].values
            mats.append(confusion_matrix(y_true, pred, labels=[0, 1]))
        cms[arm] = np.stack(mats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = ['Non-TDE', 'TDE']
    for ax, arm in zip(axes, ['Without SOM', 'With SOM']):
        m, s = cms[arm].mean(0), cms[arm].std(0)
        pct  = m / m.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap='Greens', vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{m[i,j]:.1f} +/- {s[i,j]:.1f}\n({pct[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=12,
                        color='white' if pct[i, j] > 55 else 'black')
        tn, fp, fn, tp = m.ravel()
        ax.set_title(f'{arm}\nrecall {tp/(tp+fn):.3f}   precision {tp/(tp+fp):.3f}',
                     fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(f'Pooled {N_SPLITS}-fold CV, mean +/- sd over '
                 f'{cms["With SOM"].shape[0]} repeats '
                 f'({int(cms["With SOM"][0].sum())} objects, {int(y_tde_pool.sum())} TDEs)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'cv_confusion_matrices.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    return cms


S1_CELLS = ['s1_tn_agn_rejected', 's1_fp_agn_passed',
            's1_fn_transient_rejected', 's1_tp_transient_passed',
            's1_tde_kept', 's1_tde_lost']


def stage1_summary(folds):
    """Stage-1 confusion matrix per seed (5 folds summed, so every object once)
    and its mean +/- sd over seeds, per arm (the With-SOM stage 1 also sees the
    SOM class fractions). Positive = transient (TDE + SN), negative = AGN."""
    if not set(S1_CELLS) <= set(folds.columns):
        log('stage-1 summary skipped: run predates the stage-1 confusion columns')
        return None
    per = folds.groupby(['arm', 'repeat'])[S1_CELLS].sum().reset_index()
    per['agn_rejection']    = per.s1_tn_agn_rejected / (per.s1_tn_agn_rejected + per.s1_fp_agn_passed)
    per['transient_recall'] = per.s1_tp_transient_passed / (per.s1_tp_transient_passed + per.s1_fn_transient_rejected)
    per['transient_precision'] = per.s1_tp_transient_passed / (per.s1_tp_transient_passed + per.s1_fp_agn_passed)
    per['F1'] = 2 * per.transient_precision * per.transient_recall / (per.transient_precision + per.transient_recall)
    per['tde_kept_frac'] = per.s1_tde_kept / (per.s1_tde_kept + per.s1_tde_lost)
    per.to_csv(os.path.join(TAB_DIR, 'stage1_per_seed.csv'), index=False)
    cols = S1_CELLS + ['agn_rejection', 'transient_recall', 'transient_precision', 'F1', 'tde_kept_frac']
    agg = per.groupby('arm')[cols].agg(['mean', 'std'])
    agg.to_csv(os.path.join(TAB_DIR, 'stage1_mean.csv'))

    log(f'\nSTAGE 1 (transient vs AGN), mean +/- sd over {per.repeat.nunique()} seeds')
    for arm in ['Without SOM', 'With SOM']:
        if arm not in agg.index:
            continue
        a = agg.loc[arm]
        f = lambda c: f"{a[(c, 'mean')]:.1f} +/- {a[(c, 'std')]:.1f}"
        log(f'  {arm:11s} AGN rejected {f("s1_tn_agn_rejected")}  AGN passed {f("s1_fp_agn_passed")}  '
            f'transients rejected {f("s1_fn_transient_rejected")}  passed {f("s1_tp_transient_passed")}')
        log(f'  {"":11s} TDEs kept {f("s1_tde_kept")}  lost {f("s1_tde_lost")}   '
            f"AGN rejection {a[('agn_rejection', 'mean')]:.3f}  transient recall "
            f"{a[('transient_recall', 'mean')]:.3f}  F1 {a[('F1', 'mean')]:.3f}")

    arms = [a for a in ['Without SOM', 'With SOM'] if a in agg.index]
    fig, axes = plt.subplots(1, len(arms), figsize=(6.5 * len(arms), 5.8), squeeze=False)
    labels = ['AGN', 'Transient']
    for ax, arm in zip(axes[0], arms):
        a = agg.loc[arm]
        m = np.array([[a[(S1_CELLS[0], 'mean')], a[(S1_CELLS[1], 'mean')]],
                      [a[(S1_CELLS[2], 'mean')], a[(S1_CELLS[3], 'mean')]]])
        sd = np.array([[a[(S1_CELLS[0], 'std')], a[(S1_CELLS[1], 'std')]],
                       [a[(S1_CELLS[2], 'std')], a[(S1_CELLS[3], 'std')]]])
        pct = m / m.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap='Greens', vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{m[i,j]:.1f} +/- {sd[i,j]:.1f}\n({pct[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=12,
                        color='white' if pct[i, j] > 55 else 'black')
        ax.set_title(f"{arm}\nAGN rejection {a[('agn_rejection', 'mean')]:.3f}   "
                     f"transient recall {a[('transient_recall', 'mean')]:.3f}", fontsize=11,
                     fontweight='bold', pad=10)
        ax.set_xticks([0, 1], ['rejected\n(called AGN)', 'passed to\nstage 2/3'])
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel(f"TDEs kept {a[('s1_tde_kept', 'mean')]:.1f} +/- {a[('s1_tde_kept', 'std')]:.1f}"
                      f"   lost {a[('s1_tde_lost', 'mean')]:.1f} +/- {a[('s1_tde_lost', 'std')]:.1f}",
                      fontsize=11)
        ax.set_ylabel('True')
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f'Stage 1, pooled {N_SPLITS}-fold CV, mean +/- sd over '
                 f'{per.repeat.nunique()} seeds', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stage1_confusion_matrix.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    return per


# ----------------------------------------------------------------------------
# 8 - Run
# ----------------------------------------------------------------------------
def run(pool_mags, pool_ids, y_pool, pool_strata, only_repeat=None, test=None):
    """
    Run the outer CV. With only_repeat=N, run just that repeat and write its
    results as a self-contained repeat_NN.pkl, so repeats can be farmed out to
    independent SLURM tasks and merged afterwards by merge_repeats.py.

    The splits are still drawn from the full RepeatedStratifiedKFold stream and
    the unwanted ones skipped, which costs nothing (generating a split is just
    an index shuffle) and keeps a sharded run bit-identical to a serial one.
    """
    n_repeats = max(N_REPEATS, only_repeat or 0)
    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                  random_state=0)
    fold_rows, oof_store, done = [], {arm: {} for arm in configs}, []
    # --test: per repeat and arm, the sum of the 5 fold models' scores and their
    # TDE votes for every test object (plus stage-1 pass votes)
    test_store = {}
    if test is not None:
        test_mags, test_ids = test
    t_all = time.time()
    n_done = 0
    n_todo = N_SPLITS * (1 if only_repeat else N_REPEATS)

    for it, (tr_idx, va_idx) in enumerate(rkf.split(np.zeros(len(pool_ids)),
                                                    pool_strata)):
        rep, fold = it // N_SPLITS + 1, it % N_SPLITS + 1
        if only_repeat and rep != only_repeat:
            continue
        if rep > N_REPEATS and not only_repeat:
            break
        seed = 1000 + rep                   # one seed per repeat, shared by its folds
        t0 = time.time()

        mags_tr = [pool_mags[i] for i in tr_idx]
        mags_va = [pool_mags[i] for i in va_idx]

        # inner: out-of-fold SOM features for the training rows
        tsf = oof_som_features(mags_tr, pool_ids[tr_idx], y_pool[tr_idx], seed=seed)

        # outer: one SOM on the capped training portion, scores the held-out fold
        t_som = time.time()
        sub = cap_per_class(y_pool[tr_idx], cap=CAP, seed=seed)
        som, maps, _ = fit_som_maps([mags_tr[i] for i in sub], y_pool[tr_idx][sub],
                                    seed=seed)
        vsf = apply_som_maps(som, maps, mags_va, pool_ids[va_idx])
        note(f'  som {SOM_FOLDS+1}/{SOM_FOLDS+1}  {len(sub):4d} curves  '
             f'{time.time()-t_som:5.0f}s   (SOM total {time.time()-t0:.0f}s)')

        train_tab, val_tab = build_table(tsf), build_table(vsf)
        test_tab = None
        if test is not None:
            test_tab = build_test_table(apply_som_maps(som, maps, test_mags, test_ids))
        if it == 0 or fold == 1:
            note(f'  table {train_tab.shape[0]}x{train_tab.shape[1]} train, '
                 f'{val_tab.shape[0]} val')

        line = f'f{fold} '
        for name, cfg in configs.items():
            t_arm = time.time()
            s1  = run_stage1(train_tab, val_tab, cfg['s1_drop'], seed=seed, test_df=test_tab)
            if DUMP_TABLES:
                d = os.path.join(TAB_DIR, 'dump'); os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'r{rep:02d}_f{fold}_{name.replace(" ", "")}.pkl'),
                          'wb') as fh:
                    pickle.dump({'repeat': rep, 'fold': fold, 'arm': name, 'seed': seed,
                                 'train_tab': train_tab, 'val_tab': val_tab,
                                 'train_mask': s1['train_mask'], 'val_mask': s1['val_mask'],
                                 's2_drop': cfg['s2_drop'],
                                 'pass_through': pass_through_cols}, fh)
            s23 = run_stage23(train_tab, val_tab, s1['train_mask'], s1['val_mask'],
                              cfg['s2_drop'], seed=seed,
                              s1_scores=(s1['p_tr'], s1['p_va']) if META_S1 else None,
                              test_df=test_tab, test_mask=s1.get('test_mask'))
            if test_tab is not None:
                n_te = len(test_tab)
                sc, pe = s23['test']
                acc = test_store.setdefault(name, {}).setdefault(
                    rep, {'object_id': test_tab['object_id'].to_numpy(),
                          'score_sum': np.zeros(n_te), 'votes': np.zeros(n_te, int),
                          's1_votes': np.zeros(n_te, int), 'n_folds': 0})
                assert (acc['object_id'] == test_tab['object_id'].to_numpy()).all()
                acc['score_sum'] += sc
                acc['votes'] += pe
                acc['s1_votes'] += s1['test_mask'].astype(int)
                acc['n_folds'] += 1
            note(f'  {name:11s} F1 {s23["f1_val"]:.3f}  PR-AUC {s23["pr_auc"]:.3f}  '
                 f'{time.time()-t_arm:5.0f}s')

            oof_store[name].setdefault(rep, {}).update(
                dict(zip(val_tab['object_id'],
                         zip(s23['val_scores'], s23['val_preds']))))
            if SAVE_META:
                d = os.path.join(TAB_DIR, 'meta'); os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'r{rep:02d}_f{fold}_{name.replace(" ", "")}.pkl'),
                          'wb') as fh:
                    pickle.dump({'repeat': rep, 'fold': fold, 'arm': name, 'seed': seed,
                                 'object_id': val_tab['object_id'].to_numpy(),
                                 **s23['meta_in']}, fh)

            fold_rows.append({'repeat': rep, 'fold': fold, 'arm': name,
                              'n_feat': s23['features'],
                              'F1': s23['f1_val'], 'PR-AUC': s23['pr_auc'],
                              'precision': s23['tde_precision'],
                              'recall': s23['tde_recall'],
                              'TDEs lost': s23['tde_lost'], 'FP': s23['fp'],
                              # Generalisation: f1_train comes from out-of-fold
                              # meta predictions on the training portion, f1_val
                              # from the held-out fold. Their gap says how well
                              # the training score translates - the question is
                              # whether the SOM narrows it.
                              'F1_train': s23['f1_train'],
                              'F1_gap': s23['f1_train'] - s23['f1_val'],
                              's1_f1_train': s1['f1_train'],
                              's1_f1_val': s1['f1_val'],
                              's1_agn_rejection': s1['agn_rejection'],
                              's1_tde_lost': s1['tde_lost'],
                              # stage-1 confusion matrix on the held-out fold,
                              # positive = transient (TDE + SN), negative = AGN
                              's1_tn_agn_rejected':      int(s1['cm_val'][0, 0]),
                              's1_fp_agn_passed':        int(s1['cm_val'][0, 1]),
                              's1_fn_transient_rejected': int(s1['cm_val'][1, 0]),
                              's1_tp_transient_passed':  int(s1['cm_val'][1, 1]),
                              's1_tde_kept': s1['tde_kept'],
                              's1_threshold': s1['threshold'], **s23['coef']})
            line += f"| {name.split()[0]:7s} F1 {s23['f1_val']:.3f} "

        n_done += 1
        eta = (time.time() - t_all) / n_done * (n_todo - n_done) / 60
        note(line + f'done in {(time.time()-t0)/60:.1f} min, ETA {eta:.0f} min')
        suffix = f'_rep{only_repeat:02d}' if only_repeat else ''
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(TAB_DIR, f'cv_folds{suffix}.csv'), index=False)

        if fold == N_SPLITS:
            done.append(rep)
            # Self-contained shard: everything merge_repeats.py needs for this
            # repeat, written as soon as the repeat closes.
            with open(os.path.join(TAB_DIR, f'repeat_{rep:02d}.pkl'), 'wb') as fh:
                pickle.dump({'repeat': rep,
                             'fold_rows': [r for r in fold_rows if r['repeat'] == rep],
                             'oof': {arm: oof_store[arm].get(rep, {})
                                     for arm in oof_store},
                             'test': {arm: test_store[arm].get(rep)
                                      for arm in test_store}}, fh)
            if only_repeat:
                note(f'REPEAT {rep} COMPLETE in {(time.time()-t_all)/60:.1f} min '
                     f'-> repeat_{rep:02d}.pkl')
                continue
            repeats = pool_repeats(oof_store, done)
            repeats.to_csv(os.path.join(TAB_DIR, 'cv_repeats.csv'), index=False)
            report(repeats, tag=f'rep{rep:02d}')
            # Raw per-object scores, so the analysis can be redone without refitting.
            with open(os.path.join(TAB_DIR, 'oof_store.pkl'), 'wb') as fh:
                pickle.dump({'oof_store': oof_store, 'done': done}, fh)

    log(f'\ntotal {(time.time()-t_all)/60:.0f} min')
    run.test_store = test_store
    return pd.DataFrame(fold_rows), oof_store, done


# ----------------------------------------------------------------------------
# 8b - Seeds in parallel inside ONE job
# ----------------------------------------------------------------------------
# A SLURM array gives one job per seed, which scales but scatters the output.
# This does the same work inside a single job: joblib runs K repeats at once as
# child processes, each with its own modest n_jobs, and everything lands in one
# log. Same shards, same merge, same results - only the packaging differs.
#
# Processes, not threads: the SOM is a pure-Python MiniSom loop, so threads
# would serialise on the GIL.

_WORKER_KEYS = ['FAST', 'N_SPLITS', 'N_REPEATS', 'SOM_FOLDS', 'INCLUDE_P_TDE',
                'SOM_X', 'SOM_Y', 'SOM_EPOCHS', 'CAP', 'N_MAX', 'N_JOBS',
                'TRAIN_DIR', 'VAL_DIR', 'OUT', 'FIG_DIR', 'TAB_DIR', 'PROGRESS',
                'STAGE1_PCT', 'STAGE1_MODEL', 'PEAK_LUM', 'PEAK_FILE', 'DUMP_TABLES', 'SAVE_META', 'META_SCORING', 'META_INPUTS', 'FIX_TREES', 'META_S1', 'THIRD_EXPERT', 'SECOND_EXPERT',
                'TEST', 'FINAL', 'TEST_DIR', 'DROP_BLOCK', 'ONLY_ARM', 'TEST_LOG', 'TEST_COLOUR', '_APPEND_STATS']


def _grab_cfg():
    return {k: globals()[k] for k in _WORKER_KEYS}


def _apply_cfg(cfg):
    """Re-establish the module globals inside a fresh worker process."""
    globals().update(cfg)
    globals()['VERBOSE'] = False


def _repeat_worker(rep, cfg, pool_mags, pool_ids, y_pool, pool_strata, test=None):
    """One repeat, in its own process. Returns the shard for the parent."""
    _apply_cfg(cfg)
    globals()['TAG'] = f'r{rep:02d} '
    load_inputs()        # cheap, and avoids pickling the feature table to every worker
    build_configs()
    t0 = time.time()
    folds, oof_store, _ = run(pool_mags, pool_ids, y_pool, pool_strata,
                              only_repeat=rep, test=test)
    return {'repeat': rep,
            'fold_rows': folds.to_dict('records'),
            'oof': {arm: oof_store[arm].get(rep, {}) for arm in oof_store},
            'test': {arm: d.get(rep) for arm, d in run.test_store.items()},
            'minutes': (time.time() - t0) / 60}


def run_parallel(pool_mags, pool_ids, y_pool, pool_strata, n_workers, test=None):
    from joblib import Parallel, delayed

    # The fold stream is one RepeatedStratifiedKFold(random_state=0)
    reps = list(range(FIRST_REPEAT, FIRST_REPEAT + N_REPEATS))
    cfg = _grab_cfg()
    note(f'{len(reps)} seeds, {n_workers} at a time, {N_JOBS} cores each')
    t_all = time.time()

    shards = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
        delayed(_repeat_worker)(r, cfg, pool_mags, pool_ids, y_pool, pool_strata, test)
        for r in reps)

    fold_rows, oof_store, done, test_store = [], {}, [], {}
    for sh in shards:
        rep = sh['repeat']
        done.append(rep)
        fold_rows += sh['fold_rows']
        for arm, scores in sh['oof'].items():
            oof_store.setdefault(arm, {})[rep] = scores
        with open(os.path.join(TAB_DIR, f'repeat_{rep:02d}.pkl'), 'wb') as fh:
            pickle.dump({'repeat': rep, 'fold_rows': sh['fold_rows'],
                         'oof': sh['oof'], 'test': sh.get('test', {})}, fh)
        for arm, acc in (sh.get('test') or {}).items():
            if acc is not None:
                test_store.setdefault(arm, {})[rep] = acc
        note(f'seed {rep:02d} done in {sh["minutes"]:.1f} min')

    done = sorted(done)
    write_oof_csv(oof_store)
    if test_store:
        write_test_csv(test_store)
    log(f'\nall {len(done)} seeds in {(time.time()-t_all)/60:.0f} min')
    return pd.DataFrame(fold_rows), oof_store, done


def run_final(pool_mags, pool_ids, y_pool, rep, test):
    """The fixed predictor of one seed: the whole pipeline trained once on every
    training object, exactly as a CV fold trains on its training part -
    out-of-fold SOM features for the training rows, one SOM on the capped full
    set for new objects, stage 1, experts and meta-learner with their
    thresholds from out-of-fold training scores - then applied to the test set.
    The training table is passed as the 'validation' argument too only because
    the stage functions expect one; those in-sample numbers are discarded."""
    seed = 1000 + rep
    test_mags, test_ids = test
    t0 = time.time()
    tsf = oof_som_features(pool_mags, pool_ids, y_pool, seed=seed)
    sub = cap_per_class(y_pool, cap=CAP, seed=seed)
    som, maps, _ = fit_som_maps([pool_mags[i] for i in sub], y_pool[sub], seed=seed)
    train_tab = build_table(tsf)
    test_tab = build_test_table(apply_som_maps(som, maps, test_mags, test_ids))
    store = {}
    for name, cfg in configs.items():
        s1 = run_stage1(train_tab, train_tab, cfg['s1_drop'], seed=seed, test_df=test_tab)
        s23 = run_stage23(train_tab, train_tab, s1['train_mask'], s1['val_mask'],
                          cfg['s2_drop'], seed=seed, test_df=test_tab,
                          test_mask=s1['test_mask'])
        sc, pe = s23['test']
        store[name] = {rep: {'object_id': test_tab['object_id'].to_numpy(), 'score_sum': sc,
                             'votes': pe, 's1_votes': s1['test_mask'].astype(int),
                             'n_folds': 1}}
        note(f'  {name:11s} trained on {len(train_tab)} objects, '
             f'{int(s1["test_mask"].sum())} test objects past stage 1')
    note(f'seed {rep:02d}: fixed predictor done in {(time.time()-t0)/60:.1f} min')
    return store


def _final_worker(rep, cfg, pool_mags, pool_ids, y_pool, test):
    _apply_cfg(cfg)
    globals()['TAG'] = f'r{rep:02d} '
    load_inputs()
    build_configs()
    return run_final(pool_mags, pool_ids, y_pool, rep, test)


def write_test_csv(test_store, path=None):
    """Test-set predictions, one row per (repeat, arm, object): score = mean of the
    fold models' scores, votes = how many of them called it TDE, s1_votes = how
    many passed it at stage 1. No labels in here; score_test.py adds them."""
    rows = [pd.DataFrame({'repeat': rep, 'arm': arm, 'object_id': acc['object_id'],
                          'score': acc['score_sum'] / acc['n_folds'], 'votes': acc['votes'],
                          's1_votes': acc['s1_votes'], 'n_folds': acc['n_folds']})
            for arm, by_rep in test_store.items() for rep, acc in by_rep.items()
            if acc is not None]
    path = path or os.path.join(TAB_DIR, 'test_predictions.csv.gz')
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    log(f'test-set predictions -> {path}')


def write_oof_csv(oof_store, path=None):
    """Every object's out-of-fold prediction, one row per (repeat, arm, object):
    score = stage-3 TDE probability (0 if rejected at stage 1), pred = call at
    that fold's tuned threshold. Enough to build seed ensembles later."""
    rows = [(rep, arm, oid, sc, pr)
            for arm, by_rep in oof_store.items()
            for rep, objs in by_rep.items()
            for oid, (sc, pr) in objs.items()]
    df = pd.DataFrame(rows, columns=['repeat', 'arm', 'object_id', 'score', 'pred'])
    df['SpecType'] = df.object_id.map(spec_map)
    path = path or os.path.join(TAB_DIR, 'oof_predictions.csv')
    df.sort_values(['arm', 'repeat', 'object_id']).to_csv(path, index=False)
    log(f'out-of-fold predictions -> {path} ({len(df)} rows)')


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='results',
                   help='output directory (figures/ and tables/ go inside)')
    p.add_argument('--fast', action='store_true',
                   help='500/250 trees instead of 3000/700, roughly 2x faster')
    p.add_argument('--splits', type=int, default=N_SPLITS, help='outer folds per repeat')
    p.add_argument('--repeats', type=int, default=N_REPEATS, help='number of repeats')
    p.add_argument('--som-folds', type=int, default=SOM_FOLDS)
    p.add_argument('--som-epochs', type=int, default=SOM_EPOCHS)
    p.add_argument('--som-grid', type=int, default=SOM_X, help='SOM is NxN')
    p.add_argument('--cap', type=int, default=CAP, help='per-class cap before each SOM fit')
    p.add_argument('--no-p-tde', action='store_true',
                   help='drop the raw single-node TDE fraction from stage 2')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='n_jobs handed to the estimators (default -1, all cores)')
    p.add_argument('--train-dir', default=TRAIN_DIR)
    p.add_argument('--val-dir', default=VAL_DIR)
    p.add_argument('--cache', default='lightcurve_cache.npz',
                   help='npz cache for the pooled light curves ("" to disable)')
    p.add_argument('--som-append', default='', metavar='COLS',
                   help='comma-separated columns of colour_features.csv to append '
                        'to the SOM input, e.g. "cri_p50", or with a share each, '
                        'e.g. "cri_p50:0.30,cgr_p20:0.20"')
    p.add_argument('--som-append-share', type=float, default=0.30, metavar='F',
                   help='share of SOM distance variance given to the appended '
                        'block (default 0.30)')
    p.add_argument('--stage1-pct', type=float, default=5, metavar='P',
                   help='stage-1 threshold = P-th percentile of training TDE scores '
                        '(default 5; lower keeps more TDEs and more AGN)')
    p.add_argument('--peak-lum', action='store_true',
                   help='add peak absolute magnitudes M_peak_u..y (from '
                        'peak_flux_train.csv) to stage 2/3')
    p.add_argument('--drop-block', choices=['feets', 'fits', 'color', 'z'], default=None,
                   help='feature-block ablation: remove this block from every stage')
    p.add_argument('--only-arm', choices=['with'], default=None,
                   help='run only the With-SOM arm (ablation runs)')
    p.add_argument('--final', action='store_true',
                   help='instead of CV: per seed, train the whole pipeline once on '
                        'all training objects and score the test set (the fixed '
                        'predictors); writes tables/test_predictions.csv.gz')
    p.add_argument('--test', action='store_true',
                   help='every fold model also scores the test set ('
                        'stacked_curves_short_test + *_test feature files); '
                        'writes tables/test_predictions.csv.gz, no labels used')
    p.add_argument('--second-expert', choices=['rf', 'brf'], default='rf',
                   help='second stage-2 expert: RandomForest (default) or a balanced '
                        'random forest (imblearn)')
    p.add_argument('--third-expert', choices=['et', 'lgbm', 'brf'], default='et',
                   help='third stage-2 expert: ExtraTrees (default), LightGBM or a '
                        'balanced random forest')
    p.add_argument('--meta-s1-score', action='store_true',
                   help='pass the stage-1 transient score to the meta-learner')
    p.add_argument('--fix-trees', action='store_true',
                   help='stage-2 RF/ET: drop class_weight (sample weights already '
                        'balance) and use min_samples_leaf=3')
    p.add_argument('--meta-inputs', choices=['all', 'ptde'], default='all',
                   help='meta-learner expert inputs: all 12 class probabilities '
                        '(default) or only the three P(TDE); Z and P_AGN are always passed')
    p.add_argument('--meta-scoring', default=None, metavar='SCORER',
                   help='choose the meta-learner C by this sklearn scorer, e.g. '
                        'average_precision (default: log-loss)')
    p.add_argument('--save-meta', action='store_true',
                   help='save every fold\'s meta-learner inputs to tables/meta/')
    p.add_argument('--dump-tables', action='store_true',
                   help='save every fold\'s feature tables and stage-1 masks to '
                        'tables/dump/ for stack_lab.py')
    p.add_argument('--stage1-model', choices=['rf', 'xgb'], default='rf',
                   help='stage-1 transient-vs-AGN classifier (default rf)')
    p.add_argument('--first-repeat', type=int, default=1, metavar='N',
                   help='first repeat to run with --parallel-repeats, e.g. 21 '
                        'for fresh seeds 21..20+repeats')
    p.add_argument('--som-window', default=None, metavar='LO,HI',
                   help='restrict the SOM input to these days from peak, e.g. '
                        '"0,100". Default: the whole -150..200d curve.')
    p.add_argument('--parallel-repeats', type=int, default=1, metavar='K',
                   help='run K seeds at once inside this one job (default 1, '
                        'serial). Each worker gets --n-jobs cores, so keep '
                        'K * n_jobs near the cores you asked SLURM for.')
    p.add_argument('--progress', default=None, metavar='FILE',
                   help='shared progress log every task appends to '
                        '(default <out>/progress.log; "" to disable)')
    p.add_argument('--only-repeat', type=int, default=None, metavar='N',
                   help='run only repeat N and write tables/repeat_NN.pkl. '
                        'One SLURM array task per repeat; merge_repeats.py '
                        'pools them into the final tables and figures.')
    return p.parse_args()


def main():
    global FAST, N_SPLITS, N_REPEATS, SOM_FOLDS, INCLUDE_P_TDE
    global SOM_X, SOM_Y, SOM_EPOCHS, CAP, N_JOBS, TRAIN_DIR, VAL_DIR
    global OUT, FIG_DIR, TAB_DIR, PROGRESS, TAG, VERBOSE, SOM_WINDOW
    global SOM_APPEND_COLS, SOM_APPEND_SHARE, SOM_APPEND_SHARES, FIRST_REPEAT, STAGE1_PCT
    global STAGE1_MODEL, PEAK_LUM, DUMP_TABLES, SAVE_META, META_SCORING, META_INPUTS, FIX_TREES, META_S1, THIRD_EXPERT, TEST, FINAL, SECOND_EXPERT, DROP_BLOCK, ONLY_ARM

    a = parse_args()
    VERBOSE = a.only_repeat is None
    if a.som_window:
        SOM_WINDOW = tuple(float(x) for x in a.som_window.split(','))
    items = [c for c in a.som_append.split(',') if c]
    SOM_APPEND_COLS = [c.split(':')[0] for c in items]
    # "cri_p50:0.30,cgr_p20:0.20" gives each column its own share
    SOM_APPEND_SHARES = ([float(c.split(':')[1]) for c in items]
                         if items and all(':' in c for c in items) else [])
    SOM_APPEND_SHARE = a.som_append_share
    FIRST_REPEAT = a.first_repeat
    STAGE1_PCT = a.stage1_pct
    STAGE1_MODEL = a.stage1_model
    PEAK_LUM = a.peak_lum
    DUMP_TABLES = a.dump_tables
    SAVE_META = a.save_meta
    META_SCORING = a.meta_scoring
    META_INPUTS = a.meta_inputs
    FIX_TREES = a.fix_trees
    META_S1 = a.meta_s1_score
    THIRD_EXPERT = a.third_expert
    SECOND_EXPERT = a.second_expert
    TEST = a.test
    FINAL = a.final
    DROP_BLOCK = a.drop_block
    ONLY_ARM = a.only_arm
    FAST          = a.fast
    N_SPLITS      = a.splits
    N_REPEATS     = a.repeats
    SOM_FOLDS     = a.som_folds
    SOM_EPOCHS    = a.som_epochs
    SOM_X = SOM_Y = a.som_grid
    CAP           = a.cap
    INCLUDE_P_TDE = not a.no_p_tde
    N_JOBS        = a.n_jobs
    TRAIN_DIR     = a.train_dir
    VAL_DIR       = a.val_dir

    OUT     = a.out
    FIG_DIR = os.path.join(OUT, 'figures')
    TAB_DIR = os.path.join(OUT, 'tables')
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    PROGRESS = (os.path.join(OUT, 'progress.log') if a.progress is None
                else (a.progress or None))
    TAG = f'r{a.only_repeat:02d} ' if a.only_repeat else ''

    vlog(f'{N_SPLITS} folds x {N_REPEATS} repeats = {N_SPLITS*N_REPEATS} iterations')
    vlog(f'FAST={FAST}   SOM {SOM_X}x{SOM_Y}   P_TDE as feature: {INCLUDE_P_TDE}')
    vlog(f'output -> {OUT}/')
    # Per-shard name when sharding, so concurrent array tasks don't scribble
    # over each other's record of how they were invoked.
    cfg_name = f'config_rep{a.only_repeat:02d}.json' if a.only_repeat else 'config.json'
    with open(os.path.join(OUT, cfg_name), 'w') as fh:
        json.dump(vars(a), fh, indent=2)
    if a.only_repeat and not os.path.exists(os.path.join(OUT, 'config.json')):
        # merge_repeats.py reads config.json for the figure titles.
        with open(os.path.join(OUT, 'config.json'), 'w') as fh:
            json.dump(vars(a), fh, indent=2)

    preflight()
    load_inputs()
    pool_mags, pool_ids, y_pool, pool_strata = pool_light_curves(a.cache or None)
    build_configs()
    test = test_light_curves() if (TEST or FINAL) else None

    if FINAL:
        from joblib import Parallel, delayed
        reps = list(range(FIRST_REPEAT, FIRST_REPEAT + N_REPEATS))
        note(f'final models: {len(reps)} seeds, {a.parallel_repeats} at a time, '
             f'trained on all {len(pool_ids)} training objects')
        shards = Parallel(n_jobs=max(a.parallel_repeats, 1), backend='loky')(
            delayed(_final_worker)(r, _grab_cfg(), pool_mags, pool_ids, y_pool, test)
            for r in reps)
        store = {}
        for sh in shards:
            for arm, d in sh.items():
                store.setdefault(arm, {}).update(d)
        write_test_csv(store)
        return

    if a.only_repeat:
        note(f'start  {os.uname().nodename.split(".")[0]}  {a.n_jobs} cores  '
             f'{len(pool_ids)} objects  {df_features.shape[1]} features')

    if a.parallel_repeats > 1 and not a.only_repeat:
        folds, oof_store, done = run_parallel(pool_mags, pool_ids, y_pool,
                                              pool_strata, a.parallel_repeats, test=test)
        if done:
            folds.sort_values(['repeat', 'fold', 'arm']).to_csv(
                os.path.join(TAB_DIR, 'cv_folds.csv'), index=False)
            if len(configs) == 1:
                # one-arm (ablation) run: the With/Without comparison plots don't apply
                log(f'tables  -> {TAB_DIR}/')
                return
            repeats = pool_repeats(oof_store, done)
            repeats.to_csv(os.path.join(TAB_DIR, 'cv_repeats.csv'), index=False)
            report(repeats, tag=f'merged{len(done):02d}')
            confusion_figure(oof_store)
            stage1_summary(folds)
            with open(os.path.join(TAB_DIR, 'oof_store.pkl'), 'wb') as fh:
                pickle.dump({'oof_store': oof_store, 'done': done}, fh)
            log(f'figures -> {FIG_DIR}/')
            log(f'tables  -> {TAB_DIR}/')
        return

    folds, oof_store, done = run(pool_mags, pool_ids, y_pool, pool_strata,
                                 only_repeat=a.only_repeat, test=test)
    if test is not None and not a.only_repeat:
        write_test_csv({arm: d for arm, d in run.test_store.items()})

    if a.only_repeat:
        log(f'shard done. Merge with:  python merge_repeats.py --out {OUT}')
    elif done:
        confusion_figure(oof_store)
        stage1_summary(folds)
        log(f'figures -> {FIG_DIR}/')
        log(f'tables  -> {TAB_DIR}/')
    else:
        log('no repeat completed, skipping the confusion-matrix figure')


if __name__ == '__main__':
    main()
