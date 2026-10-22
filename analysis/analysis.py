"""Full re-analysis of the Protocol-2 internal results for the manuscript.
Inputs: see paths.py (Zenodo predictions + your rebuilt cohort CSV). Outputs: analysis/out/*.csv, analysis/out/numbers.json
Published copies of every output are in results/manuscript_numbers/internal/."""
import os, json, re, glob, itertools
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import R, RESULTS_DIR, SEED_REGISTRY, load_predictions, load_summary
from scipy import stats
from scipy.stats import rankdata

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
CONF = 'a9642dff9011'; OLD = 'b6253a0a3fe1'
V = ['base', 'paag', 'csm', 'paag_csm']
LAB = {'base': 'Base', 'paag': 'PAAG', 'csm': 'CSM', 'paag_csm': 'Full'}
reg = json.load(open(SEED_REGISTRY))
SEEDS = reg['seeds'][:20]
N = {}


def auc(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------- fast DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N_ = len(x); T = np.zeros(N_); i = 0
    while i < N_:
        j = i
        while j < N_ and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N_); T2[J] = T; return T2


def delong(y, s1, s2):
    y = np.asarray(y).astype(bool); X = np.vstack([s1, s2])
    pos, neg = X[:, y], X[:, ~y]; m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(z) for z in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return d, se, 2 * stats.norm.sf(abs(d / se))


def parse(fn):
    if fn.startswith('CheXpert'):
        p = fn.split('_'); return 'CheXpert', p[1], int(p[2]), p[3][0], p[4]
    p = fn.split('_'); return 'ChestX-ray8', 'nih' + p[1], int(p[2]), p[3][0], p[4]


def load(cfg, v, s):
    return load_predictions(cfg, v, s, folder_labels=(cfg == OLD))


def summ(cfg, v, s, dev=False):
    return load_summary(('dev_' if dev else '') + cfg, v, s)


# ---------- test-set description
d0 = load(CONF, 'base', 42)
meta = pd.DataFrame([parse(f) for f in d0['test_index']], columns=['source', 'pid', 'age', 'sex', 'view'])
meta['y'] = d0['test_y']; meta['y_image'] = d0['test_y_image']; meta['y_report'] = d0['test_y_report']
vmeta = pd.DataFrame([parse(f) for f in d0['val_index']], columns=['source', 'pid', 'age', 'sex', 'view'])
vmeta['y'] = d0['val_y']
for nm, m in [('test', meta), ('val', vmeta)]:
    N[f'{nm}_n'] = len(m); N[f'{nm}_pos'] = int((m.y == 1).sum()); N[f'{nm}_neg'] = int((m.y == 0).sum())
    N[f'{nm}_masked'] = int((m.y == -1).sum()); N[f'{nm}_patients'] = int(m.pid.nunique())
    k = m.y >= 0
    N[f'{nm}_scored_patients'] = int(m.loc[k, 'pid'].nunique())
    for src in ['CheXpert', 'ChestX-ray8']:
        mm = m[m.source == src]
        N[f'{nm}_{src}_n'] = len(mm); N[f'{nm}_{src}_pos'] = int((mm.y == 1).sum()); N[f'{nm}_{src}_neg'] = int((mm.y == 0).sum())
        N[f'{nm}_{src}_masked'] = int((mm.y == -1).sum())
    for vw in ['AP', 'PA']:
        mm = m[(m.view == vw) & k]
        N[f'{nm}_{vw}_scored'] = len(mm); N[f'{nm}_{vw}_pos'] = int((mm.y == 1).sum())
    N[f'{nm}_male_frac_scored'] = float((m.loc[k, 'sex'] == 'M').mean())
    N[f'{nm}_age_median_scored'] = float(m.loc[k, 'age'].median())
    N[f'{nm}_age_iqr_scored'] = [float(m.loc[k, 'age'].quantile(.25)), float(m.loc[k, 'age'].quantile(.75))]
    N[f'{nm}_ap_frac_scored'] = float((m.loc[k, 'view'] == 'AP').mean())
# characteristics table for scored (consensus-labelled) images by split and label
rows = []
for nm, m in [('Validation', vmeta), ('Test', meta)]:
    for lab, sub in [('all scored', m[m.y >= 0]), ('pneumonia', m[m.y == 1]), ('non-pneumonia', m[m.y == 0]), ('masked', m[m.y == -1])]:
        rows.append({'split': nm, 'group': lab, 'n_images': len(sub), 'n_patients': sub.pid.nunique(),
                     'age_median': sub.age.median(), 'age_q1': sub.age.quantile(.25), 'age_q3': sub.age.quantile(.75),
                     'male_pct': 100 * (sub.sex == 'M').mean(), 'AP_pct': 100 * (sub.view == 'AP').mean(),
                     'CheXpert_pct': 100 * (sub.source == 'CheXpert').mean()})
pd.DataFrame(rows).to_csv(f'{OUT}/cohort_scored.csv', index=False)

# ---------- load all confirm runs
runs = {}
for v in V:
    for s in SEEDS:
        d = load(CONF, v, s); assert np.array_equal(d['test_index'], d0['test_index'])
        runs[(v, s)] = d
y = meta.y.values; k = y >= 0; yk = y[k]
pid_k = meta.pid.values[k]
cxp = (meta.source == 'CheXpert').values
per = pd.read_csv(f'{RESULTS_DIR}/confirmatory/per_run_metrics.csv')
# recompute and check primary AUC
chk = []
for (v, s), d in runs.items():
    a = auc(yk, d['test_p'][k]); r0 = per[(per.variant == v) & (per.seed == s)].auc.iloc[0]
    chk.append(abs(a - r0))
N['max_abs_diff_recomputed_auc'] = float(max(chk))

# ---------- per-run extra metrics
ext = []
for (v, s), d in runs.items():
    p = d['test_p']; row = {'variant': v, 'seed': s}
    for vw in ['AP', 'PA']:
        mk = k & (meta.view.values == vw); row[f'auc_{vw}'] = auc(y[mk], p[mk])
    mk = k & ~cxp; row['auc_nih'] = auc(y[mk], p[mk])
    # report-label AUC (CheXbert positive vs negative/not mentioned; NIH exact)
    yr = meta.y_report.values; mr = np.isin(yr, [0, 1, 9]); row['auc_report_label'] = auc((yr[mr] == 1).astype(int), p[mr])
    row['auc_image_label_all'] = auc(meta.y_image.values, p)
    # validation-fixed threshold at specificity 0.80
    yv, pv = d['val_y'], d['val_p']
    thr = np.quantile(pv[yv == 0], 0.80); row['thr'] = thr
    pr = p >= thr
    row['sens'] = pr[y == 1].mean(); row['spec'] = 1 - pr[y == 0].mean()
    row['ppv'] = (pr & (y == 1)).sum() / max((pr & (y >= 0)).sum(), 1)
    row['npv'] = ((~pr) & (y == 0)).sum() / max(((~pr) & (y >= 0)).sum(), 1)
    # calibration: raw / Platt brier, calibration slope/intercept of Platt on test
    a_, b_ = summ(CONF, v, s)['platt_coef'], summ(CONF, v, s)['platt_intercept']
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)); pc = 1 / (1 + np.exp(-(a_ * z + b_)))
    row['brier_raw'] = np.mean((p[k] - yk) ** 2); row['brier_platt'] = np.mean((pc[k] - yk) ** 2)
    # ECE (10 equal-width bins) after Platt
    bins = np.clip((pc[k] * 10).astype(int), 0, 9)
    row['ece_platt'] = sum(abs(pc[k][bins == b].mean() - yk[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    row['mean_pred_raw'] = p[k].mean(); row['mean_pred_platt'] = pc[k].mean()
    ext.append(row)
ext = pd.DataFrame(ext)
per2 = per.merge(ext, on=['variant', 'seed'], suffixes=('', '_re'))
per2.to_csv(f'{OUT}/per_run_full.csv', index=False)
N['prevalence_scored_test'] = float(yk.mean()); N['brier_reference'] = float(yk.mean() * (1 - yk.mean()))

desc = per2.groupby('variant').agg(['mean', 'std']).reindex(V)
desc.to_csv(f'{OUT}/per_variant_mean_sd.csv')

# ---------- seed-level paired analysis, MDD, TOST
from scipy.stats import nct, t as tdist


def mdd(sd, n, alpha=.05, power=.8):
    df = n - 1; tc = tdist.ppf(1 - alpha / 2, df)
    lo, hi = 0, 10 * sd
    for _ in range(100):
        mid = (lo + hi) / 2; ncp = mid / (sd / np.sqrt(n))
        pw = 1 - nct.cdf(tc, df, ncp) + nct.cdf(-tc, df, ncp)
        lo, hi = (mid, hi) if pw < power else (lo, mid)
    return hi


def n_needed(sd, delta, alpha=.05, power=.8):
    for n in range(3, 2000):
        df = n - 1; tc = tdist.ppf(1 - alpha / 2, df); ncp = delta / (sd / np.sqrt(n))
        if 1 - nct.cdf(tc, df, ncp) + nct.cdf(-tc, df, ncp) >= power: return n
    return np.nan


piv = per.pivot(index='seed', columns='variant', values='auc').loc[SEEDS]
pairs = [('paag', 'base'), ('csm', 'base'), ('paag_csm', 'base'), ('paag', 'csm'), ('paag_csm', 'paag'), ('paag_csm', 'csm')]
seedrows = []
for a_, b_ in pairs:
    dd = piv[a_] - piv[b_]; sd = dd.std(ddof=1)
    tt = stats.ttest_1samp(dd, 0)
    # TOST, margin 0.01 and 0.005, seed-level
    tost = {}
    for marg in (0.01, 0.005):
        se = sd / np.sqrt(len(dd)); df = len(dd) - 1
        p_lo = tdist.sf((dd.mean() + marg) / se, df); p_hi = tdist.cdf((dd.mean() - marg) / se, df)
        tost[marg] = max(p_lo, p_hi)
    ci90 = stats.t.interval(.90, len(dd) - 1, dd.mean(), sd / np.sqrt(len(dd)))
    seedrows.append({'contrast': f'{LAB[a_]} - {LAB[b_]}', 'mean': dd.mean(), 'sd': sd, 'p_t': tt.pvalue,
                     'n_pos': int((dd > 0).sum()), 'ci90_lo': ci90[0], 'ci90_hi': ci90[1],
                     'tost_p_0.01': tost[0.01], 'tost_p_0.005': tost[0.005],
                     'mdd20': mdd(sd, 20), 'mdd5': mdd(sd, 5), 'seeds_for_0.005': n_needed(sd, 0.005),
                     'seeds_for_0.0025': n_needed(sd, 0.0025)})
seedtab = pd.DataFrame(seedrows); seedtab.to_csv(f'{OUT}/seed_level.csv', index=False)

# ---------- hierarchical bootstrap (seed x patient), 90% & 95% CI (replicates P8 design, independent RNG)
rng = np.random.default_rng(20261020)
codes, inv = np.unique(pid_k, return_inverse=True)
members = [np.flatnonzero(inv == i) for i in range(len(codes))]
P = {v: np.stack([runs[(v, s)]['test_p'][k] for s in SEEDS]) for v in V}
B = 2000; boot = {p_: np.empty(B) for p_ in pairs}; boot_var = {v: np.empty(B) for v in V}
for bi in range(B):
    ss = rng.integers(0, 20, 20); idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
    yy = yk[idx]
    if yy.min() == yy.max(): continue
    A = {v: np.array([auc(yy, P[v][si][idx]) for si in np.unique(ss)]) for v in V}
    cnt = np.array([(ss == si).sum() for si in np.unique(ss)])
    for v in V: boot_var[v][bi] = np.average(A[v], weights=cnt)
    for a_, b_ in pairs: boot[(a_, b_)][bi] = np.average(A[a_] - A[b_], weights=cnt)
hb = []
for a_, b_ in pairs:
    bd = boot[(a_, b_)]
    hb.append({'contrast': f'{LAB[a_]} - {LAB[b_]}', 'mean': (piv[a_] - piv[b_]).mean(),
               'ci95_lo': np.percentile(bd, 2.5), 'ci95_hi': np.percentile(bd, 97.5),
               'ci90_lo': np.percentile(bd, 5), 'ci90_hi': np.percentile(bd, 95),
               'p_boot': min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))})
hb = pd.DataFrame(hb); hb.to_csv(f'{OUT}/hier_boot.csv', index=False)
hbv = pd.DataFrame([{'variant': v, 'mean_auc': piv[v].mean(), 'ci95_lo': np.percentile(boot_var[v], 2.5),
                     'ci95_hi': np.percentile(boot_var[v], 97.5)} for v in V]); hbv.to_csv(f'{OUT}/hier_boot_variant.csv', index=False)

# ---------- per-seed DeLong (image level) and patient-cluster bootstrap SE, variance decomposition
dl = []
for s in SEEDS:
    for a_, b_ in pairs:
        dlt, se, p = delong(yk, runs[(a_, s)]['test_p'][k], runs[(b_, s)]['test_p'][k])
        dl.append({'seed': s, 'contrast': f'{LAB[a_]} - {LAB[b_]}', 'd': dlt, 'se': se, 'p': p})
dl = pd.DataFrame(dl); dl.to_csv(f'{OUT}/delong_per_seed.csv', index=False)
dls = dl.groupby('contrast').agg(n_sig=('p', lambda x: int((x < .05).sum())), n_sig_pos=('d', 'size'), med_se=('se', 'median')).reset_index()
dls['n_sig_favA'] = dl.groupby('contrast').apply(lambda g: int(((g.p < .05) & (g.d > 0)).sum())).values
dls['n_sig_favB'] = dl.groupby('contrast').apply(lambda g: int(((g.p < .05) & (g.d < 0)).sum())).values
dls.to_csv(f'{OUT}/delong_summary.csv', index=False)
# test-sampling SE of a single model's AUC (patient bootstrap) vs between-seed SD
rng2 = np.random.default_rng(7)
pb = []
for _ in range(500):
    idx = np.concatenate([members[i] for i in rng2.integers(0, len(members), len(members))])
    pb.append(auc(yk[idx], runs[('base', 42)]['test_p'][k][idx]))
N['patient_boot_se_single_model_auc'] = float(np.std(pb))
N['patient_boot_ci_base42'] = [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]
N['between_seed_sd'] = {v: float(piv[v].std(ddof=1)) for v in V}
N['auc_range'] = {v: [float(piv[v].min()), float(piv[v].max())] for v in V}

# ---------- seed ensembles
ens = {}
for v in V:
    pe = P[v].mean(0); ens[v] = auc(yk, pe)
pe_all = np.mean([P[v].mean(0) for v in V], 0); ens['all_80'] = auc(yk, pe_all)
N['ensemble_auc'] = ens
# ensemble of the first 5 seeds only (for comparison)
N['ensemble5_auc'] = {v: auc(yk, P[v][:5].mean(0)) for v in V}

# ---------- decision churn at the validation-fixed threshold (spec 0.80): reseeding vs swapping modules
dec = {}
for (v, s), d in runs.items():
    thr = np.quantile(d['val_p'][d['val_y'] == 0], 0.80); dec[(v, s)] = (d['test_p'][k] >= thr)
within = []
for v in V:
    for s1, s2 in itertools.combinations(SEEDS, 2):
        within.append({'variant': v, 'dis': (dec[(v, s1)] != dec[(v, s2)]).mean()})
between = []
for s in SEEDS:
    for a_, b_ in pairs:
        between.append({'contrast': f'{LAB[a_]} - {LAB[b_]}', 'dis': (dec[(a_, s)] != dec[(b_, s)]).mean()})
w = pd.DataFrame(within); bt = pd.DataFrame(between)
N['churn_within_variant_reseed'] = {v: [float(w[w.variant == v].dis.mean()), float(w[w.variant == v].dis.min()), float(w[w.variant == v].dis.max())] for v in V}
N['churn_within_all_mean'] = float(w.dis.mean())
N['churn_between_same_seed'] = {c: float(g.dis.mean()) for c, g in bt.groupby('contrast')}
N['churn_between_all_mean'] = float(bt.dis.mean())


# McNemar (continuity-corrected / exact) per seed for module swap, and for reseeding (base s vs base next seed)
def mcn(a, b, yy):
    ca, cb = a == yy, b == yy; b1 = int((ca & ~cb).sum()); c1 = int((~ca & cb).sum())
    if b1 + c1 < 25: return stats.binomtest(b1, b1 + c1, .5).pvalue
    return stats.chi2.sf((abs(b1 - c1) - 1) ** 2 / (b1 + c1), 1)


yb = yk.astype(bool)
mc = []
for s in SEEDS:
    for a_, b_ in pairs:
        mc.append({'kind': 'module', 'contrast': f'{LAB[a_]} - {LAB[b_]}', 'p': mcn(dec[(a_, s)], dec[(b_, s)], yb)})
for v in V:
    for s1, s2 in zip(SEEDS[:-1], SEEDS[1:]):
        mc.append({'kind': 'reseed', 'contrast': LAB[v], 'p': mcn(dec[(v, s1)], dec[(v, s2)], yb)})
mc = pd.DataFrame(mc); mc.to_csv(f'{OUT}/mcnemar.csv', index=False)
N['mcnemar_module_sig_frac'] = float((mc[mc.kind == 'module'].p < .05).mean())
N['mcnemar_reseed_sig_frac'] = float((mc[mc.kind == 'reseed'].p < .05).mean())
N['mcnemar_module_sig_by_contrast'] = {c: int((g.p < .05).sum()) for c, g in mc[mc.kind == 'module'].groupby('contrast')}
N['mcnemar_reseed_sig_by_variant'] = {c: int((g.p < .05).sum()) for c, g in mc[mc.kind == 'reseed'].groupby('contrast')}

# ---------- training dynamics
def hist_df(cfg, v, s, dev=False):
    h = summ(cfg, v, s, dev)['history']; df = pd.DataFrame(h); df['variant'] = v; df['seed'] = s; return df


H6 = pd.concat([hist_df(CONF, v, s) for v in V for s in SEEDS]); H6['schedule'] = '6-epoch (confirmatory)'
H6.to_csv(f'{OUT}/hist_confirm.csv', index=False)
H20old = pd.concat([hist_df(OLD, v, s) for v in V for s in [42, 101, 2024, 7]]); H20old['schedule'] = '20-epoch, folder labels'
H20old.to_csv(f'{OUT}/hist_20ep_folderlabels.csv', index=False)
H20dev = pd.concat([hist_df('80e091a5af19', v, 42, dev=True) for v in V]); H20dev['schedule'] = '20-epoch, consensus labels (dev)'
H20dev.to_csv(f'{OUT}/hist_20ep_dev.csv', index=False)
devruns = {}
for h in ['80e091a5af19', 'f0dc9ac04488', 'a4cb9e818f2b', 'b029ad76da3c', 'a9642dff9011', '6634879fc7bd', '4ba5b086e734',
          'c04011106648', 'ad1682044392', '14fbd3c247b8']:
    try:
        devruns[h] = load_summary(f'dev_{h}', 'base', 42)
    except FileNotFoundError:
        pass
mf = {h: json.load(open(f)) for h, f in []}


def dyn_summary(H, label):
    g = H.groupby(['variant', 'seed'])
    rows = []
    for (v, s), df in g:
        df = df.sort_values('epoch'); bi = df.val_auc_ema.idxmax() if False else df.val_auc_ema.values.argmax()
        best = df.val_auc_ema.values[bi]; be = int(df.epoch.values[bi])
        rows.append({'schedule': label, 'variant': v, 'seed': s, 'best_epoch': be, 'epochs_run': int(df.epoch.max()),
                     'best_val_auc': best, 'last_val_auc': df.val_auc_ema.values[-1],
                     'drop_best_to_last': best - df.val_auc_ema.values[-1],
                     'auc_ep1': df.val_auc_ema.values[0],
                     'frac_of_gain_by_ep1': np.nan,
                     'train_loss_at_best': df.train_loss_pneu.values[bi], 'train_loss_last': df.train_loss_pneu.values[-1],
                     'trainloss_drop_after_best_pct': 100 * (1 - df.train_loss_pneu.values[-1] / df.train_loss_pneu.values[bi]),
                     'interior_peak': be < int(df.epoch.max())})
    return pd.DataFrame(rows)


D6 = dyn_summary(H6, '6-epoch'); D20 = dyn_summary(H20old, '20-epoch folder'); D20d = dyn_summary(H20dev, '20-epoch dev')
DYN = pd.concat([D6, D20, D20d]); DYN.to_csv(f'{OUT}/dynamics_per_run.csv', index=False)
N['dyn6_best_epoch_counts'] = D6.best_epoch.value_counts().sort_index().to_dict()
N['dyn6_epochs_run_counts'] = D6.epochs_run.value_counts().sort_index().to_dict()
N['dyn6_interior'] = int(D6.interior_peak.sum()); N['dyn6_n'] = len(D6)
N['dyn6_best_is_6'] = int((D6.best_epoch == 6).sum())
N['dyn6_drop_mean'] = float(D6.drop_best_to_last.mean()); N['dyn6_drop_range'] = [float(D6.drop_best_to_last.min()), float(D6.drop_best_to_last.max())]
N['dyn20_best_epoch_counts'] = D20.best_epoch.value_counts().sort_index().to_dict()
N['dyn20_epochs_run'] = D20.epochs_run.value_counts().sort_index().to_dict()
N['dyn20_drop_mean'] = float(D20.drop_best_to_last.mean()); N['dyn20_drop_range'] = [float(D20.drop_best_to_last.min()), float(D20.drop_best_to_last.max())]
N['dyn20_trainloss_drop_after_best'] = [float(D20.trainloss_drop_after_best_pct.min()), float(D20.trainloss_drop_after_best_pct.max())]
N['dyn20dev'] = D20d[['variant', 'best_epoch', 'epochs_run', 'best_val_auc', 'last_val_auc', 'train_loss_at_best', 'train_loss_last']].to_dict('records')
N['dyn6_trainloss_drop_after_best'] = [float(D6.trainloss_drop_after_best_pct.min()), float(D6.trainloss_drop_after_best_pct.max())]
# paired comparison seed 42: 20-epoch dev vs 6-epoch confirm (same labels, validation AUC)
N['seed42_20ep_vs_6ep'] = {v: [float(D20d[D20d.variant == v].best_val_auc.iloc[0]), float(D6[(D6.variant == v) & (D6.seed == 42)].best_val_auc.iloc[0])] for v in V}
# mean val AUC curve by epoch for 6-epoch runs (relative to each run's best)
c6 = H6.copy(); c6['rel'] = c6.groupby(['variant', 'seed']).val_auc_ema.transform(lambda x: x - x.max())
N['dyn6_mean_rel_by_epoch'] = c6.groupby('epoch').rel.mean().to_dict()
N['dyn6_mean_val_by_epoch'] = c6.groupby('epoch').val_auc_ema.mean().to_dict()
c20 = H20old.copy(); c20['rel'] = c20.groupby(['variant', 'seed']).val_auc_ema.transform(lambda x: x - x.max())
N['dyn20_mean_rel_by_epoch'] = c20.groupby('epoch').rel.mean().to_dict()
# best val AUC vs best epoch: is a later peak associated with higher AUC? (Spearman over 80 runs)
N['spearman_bestepoch_vs_valauc'] = list(map(float, stats.spearmanr(D6.best_epoch, D6.best_val_auc)))
N['spearman_bestepoch_vs_testauc'] = list(map(float, stats.spearmanr(D6.merge(per[['variant','seed','auc']], on=['variant', 'seed']).best_epoch, D6.merge(per[['variant','seed','auc']], on=['variant', 'seed']).auc)))
# dev sweep
sweep = []
for h, sj in devruns.items():
    hh = pd.DataFrame(sj['history'])
    sweep.append({'hash': h, 'best_val_auc': sj['best_val_auc_ema'], 'best_epoch': sj['best_epoch'], 'epochs_run': sj['epochs_run'],
                  'last_val_auc': hh.val_auc_ema.values[-1], 'train_hours': sj['train_hours']})
pd.DataFrame(sweep).to_csv(f'{OUT}/dev_sweep_full.csv', index=False)
N['train_hours_6ep_mean'] = float(np.mean([summ(CONF, v, s)['train_hours'] for v in V for s in SEEDS]))
N['train_hours_20ep_folder_mean'] = float(np.mean([summ(OLD, v, s)['train_hours'] for v in V for s in [42, 101, 2024, 7]]))
N['train_hours_6ep_total'] = float(np.sum([summ(CONF, v, s)['train_hours'] for v in V for s in SEEDS]))

# ---------- 20-epoch folder-label runs: test AUC on folder labels (for label-definition decomposition)
old = []
for v in (V if os.path.exists(f'{R}/{OLD}') else []):     # the 16 development runs; skipped if not downloaded
    for s in [42, 101, 2024, 7]:
        e = load(OLD, v, s); sm = summ(OLD, v, s)
        old.append({'variant': v, 'seed': s, 'test_auc_folder': auc(e['test_y'], e['test_p']), 'best_epoch': sm['best_epoch'],
                    'epochs_run': sm['epochs_run'], 'best_val_auc': sm['best_val_auc_ema']})
old = pd.DataFrame(old) if old else pd.read_csv(f'{RESULTS_DIR}/manuscript_numbers/internal/old_20ep_folder.csv')
old.to_csv(f'{OUT}/old_20ep_folder.csv', index=False)
N['old20_test_auc_folder_mean'] = old.groupby('variant').test_auc_folder.mean().reindex(V).to_dict()
N['old20_test_auc_folder_sd'] = old.groupby('variant').test_auc_folder.std().reindex(V).to_dict()
# the confirm models evaluated on folder labels, same 4 seeds
N['conf_on_folder_same4'] = per[per.seed.isin([42, 101, 2024, 7])].groupby('variant').auc_image_label_all.mean().reindex(V).to_dict()
N['test_folder_pos'] = int(meta.y_image.sum())

# ---------- localization (central-box fallback)
loc = pd.read_csv(f'{RESULTS_DIR}/localisation/central_box_first_run/per_run_localization.csv')
N['loc_reference'] = loc.reference.unique().tolist()
locs = loc.groupby('variant')[['pneumonia_cam_lung', 'pneumonia_cam_iou_lung', 'pneumonia_attn_lung']].agg(['mean', 'std']).reindex(V)
locs.to_csv(f'{OUT}/loc_summary.csv')
lp = []
for a_ in ['paag', 'csm', 'paag_csm']:
    for m in ['pneumonia_cam_lung', 'pneumonia_cam_iou_lung']:
        pv = loc.pivot(index='seed', columns='variant', values=m); dd = (pv[a_] - pv['base']).dropna()
        lp.append({'contrast': f'{LAB[a_]} - Base', 'metric': m, 'mean': dd.mean(), 'sd': dd.std(), 'n_pos': int((dd > 0).sum()),
                   'n': len(dd), 'p_t': stats.ttest_1samp(dd, 0).pvalue})
lp = pd.DataFrame(lp)
lp['p_holm'] = np.nan
for m in lp.metric.unique():
    idx = lp.index[lp.metric == m]; ps = lp.loc[idx, 'p_t'].values; o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
    lp.loc[idx, 'p_holm'] = adj
lp.to_csv(f'{OUT}/loc_contrasts.csv', index=False)

json.dump(N, open(f'{OUT}/numbers.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(N, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:6000])
print(desc[['auc', 'auc_chexpert', 'auc_AP', 'auc_PA', 'auc_nih', 'auc_image_label_all', 'auc_report_label', 'sens', 'spec', 'ppv', 'npv', 'cardio_fpr_at_spec', 'brier_raw', 'brier_platt', 'ece_platt', 'best_epoch']].round(4).T.to_string())
print(seedtab.round(4).to_string()); print(hb.round(4).to_string()); print(dls.to_string())
