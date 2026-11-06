"""
r-i color at +50 days of every object, the term cv_som.py appends to the SOM input.


Reads only the stacked curves and the training log. Test labels are never read:
every test object with a complete g, r, i grid is kept.
"""
import glob
import os

import numpy as np
import pandas as pd

TRAIN_DIRS = ['stacked_curves_short_all_train', 'stacked_curves_short_all_val']
TEST_DIR = 'stacked_curves_short_test'
T = np.linspace(-150, 200, 100)     # grid of the stacked curves, days from the peak


def cidx(a, b):
    """Color index (a - b) / (|a| + |b|) of two normalized curves, in [-1, 1].

    Returns 0 where both fluxes are essentially zero (|a| + |b| <= 1e-6).
    """
    den = np.abs(a) + np.abs(b)
    return np.where(den > 1e-6, (a - b) / np.where(den > 1e-6, den, 1.0), 0.0)


def at(curve, day):
    """Value of a curve on the grid T at the epoch closest to `day`."""
    return curve[np.argmin(np.abs(T - day))]


def cri_p50(path):
    """r-i color at +50 days from one stacked curve, or None if the grid is incomplete.

    Despite their names, the columns mjd and mag of a stacked curve hold the days
    from the peak and the normalized flux. A complete grid has 100 points in each
    of g, r and i.
    """
    d = pd.read_csv(path)
    b = {k: np.nan_to_num(g['mag'].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
         for k, g in d.groupby('band')}
    if not all(k in b and len(b[k]) == 100 for k in 'gri'):
        return None
    return at(cidx(b['r'], b['i']), 50)


def colour_table(folders, keep=None):
    """DataFrame with object_id and cri_p50 for the stacked curves in `folders`.

    Curve files are named df_<object_id>.csv. If `keep` is given, only the object
    ids it contains are used.
    """
    rows = []
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, '*.csv'))):
            oid = os.path.basename(f)[3:-4]
            if keep is not None and oid not in keep:
                continue
            c = cri_p50(f)
            if c is not None:
                rows.append({'object_id': oid, 'cri_p50': c})
    return pd.DataFrame(rows)


def main():
    """Write colour_features.csv (labeled training objects) and colour_features_test.csv."""
    labels = pd.read_csv('train_log.csv').set_index('object_id')['SpecType'].dropna()
    train = colour_table(TRAIN_DIRS, keep=set(labels.index))
    test = colour_table([TEST_DIR])
    train.to_csv('colour_features.csv', index=False)
    test.to_csv('colour_features_test.csv', index=False)
    print(f'{len(train)} training and {len(test)} test objects')


if __name__ == '__main__':
    main()
