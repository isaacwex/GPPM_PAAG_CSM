"""Exploratory follow-up (Table S8b): how much could any correction of the pneumonia score from the five
comorbidity outputs add? For each confirmatory model, a logistic regression on [logit p_raw, 5 comorbidity logits]
is fitted on VALIDATION and applied to TEST. Inputs: see paths.py (labels re-attached by position).
Output: analysis/out/csm_ceiling.csv (published copy: results/manuscript_numbers/internal/csm_ceiling.csv)."""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SEED_REGISTRY, load_predictions

H = os.path.dirname(os.path.abspath(__file__))
lg = lambda a: np.log(np.clip(a, 1e-6, 1 - 1e-6) / np.clip(1 - a, 1e-6, 1))
seeds = json.load(open(SEED_REGISTRY))['seeds'][:20]
rows = []
for v in ['base', 'paag', 'csm', 'paag_csm']:
    for s in seeds:
        d = load_predictions('a9642dff9011', v, s)
        y, yt = d['val_y'], d['test_y']; k, kt = y >= 0, yt >= 0
        X = np.column_stack([lg(d['val_p_raw'][k]), lg(d['val_com'][:, :5][k])])
        Xt = np.column_stack([lg(d['test_p_raw'][kt]), lg(d['test_com'][:, :5][kt])])
        m = LogisticRegression(max_iter=5000).fit(X, y[k]); pt = m.predict_proba(Xt)[:, 1]; pv = m.predict_proba(X)[:, 1]
        a0, a1 = roc_auc_score(yt[kt], Xt[:, 0]), roc_auc_score(yt[kt], pt)
        thr0, thr1 = np.quantile(X[y[k] == 0, 0], .8), np.quantile(pv[y[k] == 0], .8)
        r0, r1 = (Xt[:, 0] >= thr0)[yt[kt] == 1].mean(), (pt >= thr1)[yt[kt] == 1].mean()
        rows.append({'variant': v, 'seed': s, 'auc_p_raw': a0, 'auc_combined': a1, 'gain': a1 - a0, 'recall_gain': r1 - r0})
df = pd.DataFrame(rows)
os.makedirs(f'{H}/out', exist_ok=True)
df.to_csv(f'{H}/out/csm_ceiling.csv', index=False)
print(df.groupby('variant')[['gain', 'recall_gain']].agg(['mean', 'std', 'min', 'max']).round(4))
