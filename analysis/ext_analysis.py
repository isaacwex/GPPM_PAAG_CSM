"""External evaluation on PadChest (physician labels, frontal, age >= 65): 80 frozen confirmatory models."""
import os, json, itertools
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import R, RESULTS_DIR, SEED_REGISTRY, PADCHEST_MANIFEST, EXTERNAL_NPZ, load_predictions, load_summary
from scipy import stats
from scipy.stats import rankdata, t as tdist

H = os.path.dirname(os.path.abspath(__file__)); O = f'{H}/out_ext'; os.makedirs(O, exist_ok=True)
CONF = 'a9642dff9011'
man = pd.read_csv(PADCHEST_MANIFEST)
D = np.load(EXTERNAL_NPZ)
assert (D['image_id'] == man.ImageID.values).all()
V = ['base', 'paag', 'csm', 'paag_csm']; LAB = {'base': 'Base', 'paag': 'PAAG', 'csm': 'CSM', 'paag_csm': 'Full'}
SEEDS = json.load(open(SEED_REGISTRY))['seeds'][:20]
y = man.pneumonia.values.astype(int); pid = man.PatientID.astype(str).values
N = {'n': len(man), 'pos': int(y.sum()), 'patients': int(man.PatientID.nunique()), 'prev': float(y.mean()),
     'age_median': float(man.age.median()), 'age_iqr': [float(man.age.quantile(.25)), float(man.age.quantile(.75))],
     'male_pct': float((man.PatientSex_DICOM == 'M').mean() * 100), 'pa_pct': float((man.view == 'PA').mean() * 100),
     'pos_pa': int(y[man.view.values == 'PA'].sum()), 'n_pa': int((man.view == 'PA').sum())}


def auc(yy, s):
    yy = np.asarray(yy); r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def P(v, s, k='p'):
    return D[f'{v}_seed{s}__{k}'].astype(np.float64)


COM = list(D['com_names'])
strata = {'cardiomegaly': man.com_Cardiomegaly.values == 1, 'pleural': man.com_Pleural_Extrapulmonary_Volume_Disease.values == 1,
          'heart_insufficiency': man.heart_insufficiency.values == 1, 'focal': man.com_Focal_Lesion_Chronic_Change.values == 1,
          'normal': man.com_No_Finding.values == 1}
nocom = ~(strata['cardiomegaly'] | strata['pleural'] | strata['heart_insufficiency'] | strata['focal'] |
          (man.com_Weakly_Supervised_Uncertain.values == 1))
amb = man.airspace_no_pneumonia.values == 1
for k, m in strata.items():
    N[f'n_{k}'] = int(m.sum()); N[f'n_{k}_pos'] = int((m & (y == 1)).sum())
N['n_nocom'] = int(nocom.sum()); N['n_nocom_pos'] = int((nocom & (y == 1)).sum()); N['n_amb'] = int(amb.sum())

rows = []
for v in V:
    for s in SEEDS:
        p = P(v, s); r = {'variant': v, 'seed': s, 'auc': auc(y, p)}
        r['auc_PA'] = auc(y[man.view.values == 'PA'], p[man.view.values == 'PA'])
        r['auc_AP'] = auc(y[man.view.values == 'AP'], p[man.view.values == 'AP'])
        r['auc_excl_ambiguous'] = auc(y[~amb], p[~amb])
        for k in ['cardiomegaly', 'pleural', 'heart_insufficiency']:
            m = strata[k]; r[f'auc_within_{k}'] = auc(y[m], p[m])
        r['auc_within_nocom'] = auc(y[nocom], p[nocom])
        # transported operating point: internal-validation threshold at specificity 0.80
        iv = load_predictions(CONF, v, s); thr = np.quantile(iv['val_p'][iv['val_y'] == 0], 0.80)
        pr = p >= thr
        r['sens'] = pr[y == 1].mean(); r['spec'] = 1 - pr[y == 0].mean()
        for k in ['cardiomegaly', 'pleural', 'heart_insufficiency', 'normal']:
            m = strata[k] & (y == 0); r[f'fpr_{k}'] = pr[m].mean()
        r['fpr_nocom'] = pr[nocom & (y == 0)].mean()
        # internal Platt transported
        sm = load_summary(CONF, v, s)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)); pc = 1 / (1 + np.exp(-(sm['platt_coef'] * z + sm['platt_intercept'])))
        r['brier_platt'] = np.mean((pc - y) ** 2); r['mean_pred_platt'] = pc.mean()
        # comorbidity heads (external validity of what CSM relies on)
        com = D[f'{v}_seed{s}__com'].astype(np.float64)
        r['head_cardiomegaly'] = auc(man.com_Cardiomegaly.values, com[:, COM.index('Cardiomegaly')])
        r['head_pleural'] = auc(man.com_Pleural_Extrapulmonary_Volume_Disease.values, com[:, COM.index('Pleural_Extrapulmonary_Volume_Disease')])
        r['head_focal'] = auc(man.com_Focal_Lesion_Chronic_Change.values, com[:, COM.index('Focal_Lesion_Chronic_Change')])
        r['head_nofinding'] = auc(man.com_No_Finding.values, com[:, COM.index('No_Finding')])
        r['head_edema'] = auc(man.com_Weakly_Supervised_Uncertain.values, com[:, COM.index('Weakly_Supervised_Uncertain')])
        r['aux_effusion'] = auc(man.pleural_effusion.values, com[:, COM.index('aux_Pleural Effusion')])
        if v in ('csm', 'paag_csm'):
            sup = P(v, s, 'supp')
            r['supp_cardio_neg'] = sup[strata['cardiomegaly'] & (y == 0)].mean(); r['supp_normal'] = sup[strata['normal']].mean()
            r['supp_pneu'] = sup[y == 1].mean()
        rows.append(r)
per = pd.DataFrame(rows); per.to_csv(f'{O}/per_run_external.csv', index=False)
desc = per.groupby('variant').agg(['mean', 'std']).reindex(V); desc.to_csv(f'{O}/external_mean_sd.csv')
N['brier_reference'] = float(y.mean() * (1 - y.mean()))

# internal per-run AUC for correlation
pin = pd.read_csv(f'{RESULTS_DIR}/confirmatory/per_run_metrics.csv')[['variant', 'seed', 'auc']].rename(columns={'auc': 'auc_int'})
mm = per.merge(pin, on=['variant', 'seed'])
N['spearman_int_ext'] = list(map(float, stats.spearmanr(mm.auc_int, mm.auc)))
N['pearson_int_ext'] = list(map(float, stats.pearsonr(mm.auc_int, mm.auc)))

# paired contrasts: seed-level + hierarchical bootstrap (seed x patient)
pairs = [('paag', 'base'), ('csm', 'base'), ('paag_csm', 'base'), ('paag', 'csm'), ('paag_csm', 'paag'), ('paag_csm', 'csm')]
piv = per.pivot(index='seed', columns='variant', values='auc').loc[SEEDS]
codes, inv = np.unique(pid, return_inverse=True); members = [np.flatnonzero(inv == i) for i in range(len(codes))]
PP = {v: np.stack([P(v, s) for s in SEEDS]) for v in V}
rng = np.random.default_rng(20261022); B = 2000; boot = {p_: [] for p_ in pairs}
for bi in range(B):
    ss = rng.integers(0, 20, 20); idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
    yy = y[idx]; us, cnt = np.unique(ss, return_counts=True)
    A = {v: np.array([auc(yy, PP[v][si][idx]) for si in us]) for v in V}
    for a_, b_ in pairs: boot[(a_, b_)].append(np.average(A[a_] - A[b_], weights=cnt))
rowsc = []
for a_, b_ in pairs:
    dd = piv[a_] - piv[b_]; bd = np.array(boot[(a_, b_)]); sd = dd.std(ddof=1); se = sd / np.sqrt(20)
    tost = {m: max(tdist.sf((dd.mean() + m) / se, 19), tdist.cdf((dd.mean() - m) / se, 19)) for m in (0.005, 0.01)}
    rowsc.append({'contrast': f'{LAB[a_]} - {LAB[b_]}', 'primary': b_ == 'base', 'dAUC': dd.mean(),
                  'ci95_lo': np.percentile(bd, 2.5), 'ci95_hi': np.percentile(bd, 97.5), 'ci90_lo': np.percentile(bd, 5),
                  'ci90_hi': np.percentile(bd, 95), 'p_boot': min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean())),
                  'seeds_A_gt_B': int((dd > 0).sum()), 'p_t': stats.ttest_1samp(dd, 0).pvalue, 'tost_0.005': tost[0.005],
                  'tost_0.01': tost[0.01]})
con = pd.DataFrame(rowsc)
pr_ = con[con.primary].p_boot.values; o = np.argsort(pr_); adj = np.empty(3); run = 0
for r_, k in enumerate(o): run = max(run, (3 - r_) * pr_[k]); adj[k] = min(1, run)
con.loc[con.primary, 'p_holm'] = adj
con.to_csv(f'{O}/contrasts_external.csv', index=False)
# secondary paired contrasts with Base (seed-level t)
sec = []
for m in ['fpr_cardiomegaly', 'fpr_pleural', 'fpr_heart_insufficiency', 'fpr_normal', 'sens', 'auc_within_cardiomegaly',
          'auc_within_pleural', 'auc_excl_ambiguous', 'auc_PA', 'auc_AP']:
    pv = per.pivot(index='seed', columns='variant', values=m)
    for a_ in ['paag', 'csm', 'paag_csm']:
        dd = pv[a_] - pv['base']
        sec.append({'metric': m, 'contrast': f'{LAB[a_]} - Base', 'mean_diff': dd.mean(), 'sd': dd.std(), 'n_pos': int((dd > 0).sum()),
                    'p_t': stats.ttest_1samp(dd, 0).pvalue})
pd.DataFrame(sec).to_csv(f'{O}/secondary_external.csv', index=False)
# ensembles
N['ensemble_auc'] = {v: float(auc(y, PP[v].mean(0))) for v in V}; N['ensemble_all'] = float(auc(y, np.mean([PP[v].mean(0) for v in V], 0)))
# single-model test-sampling SE (patient bootstrap), base seed 42
rng2 = np.random.default_rng(3); pb = []
for _ in range(500):
    idx = np.concatenate([members[i] for i in rng2.integers(0, len(members), len(members))]); pb.append(auc(y[idx], PP['base'][0][idx]))
N['patient_boot_se_single'] = float(np.std(pb))
json.dump(N, open(f'{O}/numbers_ext.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print(json.dumps(N, indent=1))
cols = ['auc', 'auc_PA', 'auc_AP', 'auc_excl_ambiguous', 'auc_within_cardiomegaly', 'auc_within_pleural', 'auc_within_heart_insufficiency',
        'auc_within_nocom', 'sens', 'spec', 'fpr_cardiomegaly', 'fpr_pleural', 'fpr_heart_insufficiency', 'fpr_normal', 'fpr_nocom',
        'brier_platt', 'mean_pred_platt', 'head_cardiomegaly', 'head_pleural', 'head_focal', 'head_nofinding', 'head_edema', 'aux_effusion',
        'supp_cardio_neg', 'supp_normal', 'supp_pneu']
for c in cols:
    print(f"{c:32s}", ' | '.join(f"{desc.loc[v, (c, 'mean')]:.4f}±{desc.loc[v, (c, 'std')]:.4f}" if not np.isnan(desc.loc[v, (c, 'mean')]) else '—' for v in V))
print(con.round(4).to_string()); print(pd.DataFrame(sec).round(4).to_string())
