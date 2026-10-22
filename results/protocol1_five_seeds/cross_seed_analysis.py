#!/usr/bin/env python3
"""
Cross-seed statistical synthesis for the PAAG/CSM geriatric pneumonia ablation.

Standalone, dependency-light re-implementation of the pairwise DeLong test and
DerSimonian-Laird (HKSJ-adjusted) random-effects pooling used in the project
notebooks, operating directly on the cached test-set predictions.npz files
under <results-root>/BestModel/<variant>/[seed_<seed>/test_rescored/]predictions.npz.

Produces one row per (variant pair, seed) plus a pooled estimate across seeds,
written to cross_seed_synthesis.csv.

Usage:
    python cross_seed_analysis.py --results-root /notebooks/results_PAAG_CSM \
        --seeds 42 101 2024 7 2025 --out cross_seed_synthesis.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

VARIANTS = ['base', 'paag', 'csm', 'paag_csm']
LABELS = {'base': 'Base', 'paag': 'PAAG', 'csm': 'CSM', 'paag_csm': 'Full model'}
PAIRS = [('base', 'paag'), ('base', 'csm'), ('base', 'paag_csm'),
         ('paag', 'csm'), ('paag', 'paag_csm'), ('csm', 'paag_csm')]


def predictions_path(results_root, variant, seed, default_seed):
    base = os.path.join(results_root, 'BestModel', variant)
    if seed == default_seed:
        return os.path.join(base, 'predictions.npz')
    return os.path.join(base, f'seed_{seed}', 'test_rescored', 'predictions.npz')


def load_predictions(results_root, variant, seed, default_seed):
    path = predictions_path(results_root, variant, seed, default_seed)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {'y_true': data['y_true'].astype(int), 'y_scores': data['y_scores'].astype(float)}


def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true, scores_a, scores_b):
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    order = np.argsort(-y_true, kind='mergesort')
    m = int(np.sum(y_true))
    n = len(y_true) - m
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    preds = np.vstack([scores_a, scores_b])[:, order]
    positive, negative = preds[:, :m], preds[:, m:]

    tx = np.vstack([_midrank(positive[r]) for r in range(2)])
    ty = np.vstack([_midrank(negative[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(var) if var > 0 else np.nan
    p = 2 * scipy_stats.norm.sf(abs(diff / se)) if se and se > 0 else 1.0
    return aucs[0], aucs[1], se, p


def pool_random_effects(deltas, ses):
    d, se = np.asarray(deltas, dtype=float), np.asarray(ses, dtype=float)
    valid = se > 0
    d, se = d[valid], se[valid]
    k = len(d)
    if k < 2:
        return {'pooled': d[0] if k == 1 else np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'p': np.nan, 'p_hksj': np.nan, 'I2': np.nan, 'k': k}

    w = 1.0 / se ** 2
    fixed_effect = (w * d).sum() / w.sum()
    q_stat = (w * (d - fixed_effect) ** 2).sum()
    c_const = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q_stat - (k - 1)) / c_const) if c_const > 0 else 0.0

    w_random = 1.0 / (se ** 2 + tau2)
    pooled = (w_random * d).sum() / w_random.sum()
    se_pooled = np.sqrt(1.0 / w_random.sum())

    q_hksj = (w_random * (d - pooled) ** 2).sum() / max(k - 1, 1)
    se_hksj = np.sqrt(q_hksj / w_random.sum())

    p_normal = 2 * scipy_stats.norm.sf(abs(pooled / se_pooled)) if se_pooled > 0 else np.nan
    p_hksj = 2 * scipy_stats.t.sf(abs(pooled / se_hksj), max(k - 1, 1)) if se_hksj > 0 and k > 1 else np.nan
    i2 = max(0.0, (q_stat - (k - 1)) / q_stat) * 100 if q_stat > 0 else 0.0

    return {'pooled': pooled, 'ci_low': pooled - 1.96 * se_pooled, 'ci_high': pooled + 1.96 * se_pooled,
            'p': p_normal, 'p_hksj': p_hksj, 'I2': i2, 'k': k}


def run(results_root, seeds, default_seed, out_path):
    predictions = {}
    for variant in VARIANTS:
        for seed in seeds:
            r = load_predictions(results_root, variant, seed, default_seed)
            if r is not None:
                predictions[(variant, seed)] = r

    rows = []
    for a_, b_ in PAIRS:
        per_seed_delta, per_seed_se, per_seed_p, seeds_used = [], [], [], []
        for seed in seeds:
            if (a_, seed) not in predictions or (b_, seed) not in predictions:
                continue
            pa, pb = predictions[(a_, seed)], predictions[(b_, seed)]
            if not np.array_equal(pa['y_true'], pb['y_true']):
                raise RuntimeError(f"Label mismatch between '{a_}' and '{b_}' at seed {seed} -- "
                                    f"these predictions were not scored on the same test set.")
            auc_a, auc_b, se, p = delong_test(pa['y_true'], pa['y_scores'], pb['y_scores'])
            per_seed_delta.append(auc_a - auc_b)
            per_seed_se.append(se)
            per_seed_p.append(p)
            seeds_used.append(seed)

        if not seeds_used:
            continue

        pooled = pool_random_effects(per_seed_delta, per_seed_se)
        row = {'comparison': f'{LABELS[a_]} vs {LABELS[b_]}', 'n_seeds': len(seeds_used),
               'n_seeds_p<0.05': int(sum(p < 0.05 for p in per_seed_p if not np.isnan(p)))}
        for seed, delta, p in zip(seeds_used, per_seed_delta, per_seed_p):
            row[f'dAUC_seed{seed}'] = delta
            row[f'p_seed{seed}'] = p
        row.update({f'pooled_{k}': v for k, v in pooled.items()})
        rows.append(row)

    df = pd.DataFrame(rows).set_index('comparison')
    df.to_csv(out_path)
    print(df.round(4).to_string())
    print(f"\nSaved to {out_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True,
                        help="Path to the results_PAAG_CSM folder (contains BestModel/).")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 101, 2024, 7, 2025])
    parser.add_argument('--default-seed', type=int, default=42,
                        help="The seed whose checkpoint is cached directly under BestModel/<variant>/ "
                             "rather than BestModel/<variant>/seed_<n>/test_rescored/.")
    parser.add_argument('--out', default='cross_seed_synthesis.csv')
    args = parser.parse_args()

    run(args.results_root, args.seeds, args.default_seed, args.out)
