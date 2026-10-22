import os, json, itertools
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import R, RESULTS_DIR, SEED_REGISTRY, load_predictions, load_summary
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import rankdata

H = os.path.dirname(os.path.abspath(__file__)); O = f'{H}/out'; F = f'{H}/fig'; os.makedirs(F, exist_ok=True)
CONF = 'a9642dff9011'
V = ['base', 'paag', 'csm', 'paag_csm']; LAB = {'base': 'Base', 'paag': 'PAAG', 'csm': 'CSM', 'paag_csm': 'Full (PAAG+CSM)'}
COL = {'base': '#2a78d6', 'paag': '#eb6834', 'csm': '#1baf7a', 'paag_csm': '#6250d6'}
MK = {'base': 'o', 'paag': 's', 'csm': '^', 'paag_csm': 'D'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left'})
N = json.load(open(f'{O}/numbers.json'))
reg = json.load(open(SEED_REGISTRY)); SEEDS = reg['seeds'][:20]


def save(fig, name):
    fig.savefig(f'{F}/{name}.png', bbox_inches='tight', facecolor='white'); plt.close(fig)


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ================= Figure 1: study design
fig, ax = plt.subplots(figsize=(10.5, 6.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 100)


def box(x, y, w, h, title, body, fc='#f3f6fb', ec='#2a78d6'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1))
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=8, fontweight='bold', color=INK, va='top')
    ax.text(x + 1.2, y + h - 6.6, body, fontsize=6.9, color=INK2, va='top', linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1))


box(1, 70, 30, 28, 'Geriatric cohort (age ≥ 65)',
    'CheXpert frontal: 45,848 files\nChestX-ray8: 15,547 files\nPatient-level split 60:20:20\nTrain 36,620 · Val 12,555 · Test 12,220')
box(35, 70, 30, 28, 'Consensus pneumonia label',
    'CheXpert: image label (VisualCheXbert)\nAND report label (CheXbert) must agree;\ndisagreement → masked (−1)\nChestX-ray8: NIH label (exact)\nTest scored: 608 pos / 9,020 neg')
box(69, 70, 30, 28, 'Model variants (DenseNet-121)',
    'Base · +PAAG (spatial gate)\n+CSM (bounded suppressor)\nFull = PAAG + CSM\nShared heads: 5 comorbidity groups\n+ 13 auxiliary CheXpert findings')
box(1, 35, 46, 28, 'Development (validation set only; test never scored)',
    '20-epoch schedule (seed 42): peak at epoch 3–4, then decline\nSweep of 7 recipes (regularisation, sqrt pos-weight,\nMIMIC-CXR pretraining, 6-epoch schedule)\nPre-specified rule → frozen recipe: 6 epochs, patience 3\nConfig hash a9642dff9011', fc='#fff6f1', ec='#eb6834')
box(53, 35, 46, 28, 'Confirmation (pre-specified, frozen config)',
    'First 20 seeds of a 40-seed registry (fixed in advance)\n4 variants × 20 seeds = 80 models\nCheckpoint: best validation AUC of EMA weights\nThreshold and Platt scaling fitted on validation\nTest scored once per model', fc='#f1faf6', ec='#1baf7a')
box(1, 2, 98, 26, 'Pre-specified analysis',
    'Primary: test ROC-AUC, contrasts PAAG−Base, CSM−Base, Full−Base; seed × patient hierarchical bootstrap (2,000 resamples), Holm adjustment\n'
    'Secondary: within-CheXpert AUC; cardiomegaly false-positive rate and sensitivity at validation-fixed specificity 0.80; Brier after Platt;\n'
    'Grad-CAM localisation.  Additional (post hoc): equivalence tests (TOST, ±0.005 / ±0.01), DeLong and McNemar per seed, decision churn,\n'
    'seed ensembles, label-definition sensitivity, training dynamics of 100 runs, anatomical localisation, module activity;\n'
    'external evaluation of all 80 frozen models on 12,038 physician-labelled PadChest images (418 pneumonia)', fc='#f7f7f5', ec=INK2)
for x in (31, 65): arrow(x, 84, x + 4, 84)
arrow(84, 70, 84, 63.5); arrow(24, 70, 24, 63.5); arrow(47, 49, 53, 49); arrow(76, 35, 76, 28.5)
save(fig, 'Figure1_design')

# ================= Figure 2: training dynamics
H6 = pd.read_csv(f'{O}/hist_confirm.csv'); H20 = pd.read_csv(f'{O}/hist_20ep_folderlabels.csv'); H20d = pd.read_csv(f'{O}/hist_20ep_dev.csv')
dyn = pd.read_csv(f'{O}/dynamics_per_run.csv'); per = pd.read_csv(f'{O}/per_run_full.csv')
for h in (H6, H20, H20d):
    h['rel'] = h.groupby(['variant', 'seed']).val_auc_ema.transform(lambda x: x - x.max())
    h['loss_rel'] = h.groupby(['variant', 'seed']).train_loss_pneu.transform(lambda x: x / x.iloc[0])
fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6))
ax = axs[0, 0]
for (v, s), g in H20.groupby(['variant', 'seed']):
    ax.plot(g.epoch, g.rel, color='#9a9994', lw=.8, alpha=.6)
for (v, s), g in H20d.groupby(['variant', 'seed']):
    ax.plot(g.epoch, g.rel, color=COL[v], lw=1.6)
m = H20.groupby('epoch').rel.mean(); ax.plot(m.index, m.values, color=INK, lw=2, label='Mean, 16 runs (original labels)')
ax.plot([], [], color='#9a9994', lw=.8, label='Individual runs, original labels')
ax.plot([], [], color=COL['base'], lw=1.6, label='Consensus labels, seed 42 (4 variants)')
ax.axvline(6, color=INK2, ls=':', lw=1); ax.text(6.15, 0.0015, 'chosen budget', fontsize=6.8, color=INK2, va='bottom')
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation AUC minus run peak'); ax.set_xlim(.7, 11.3); ax.set_xticks(range(1, 12))
ax.set_title('a  20-epoch schedule (patience 6)'); ax.legend(fontsize=6.5, loc='lower left')
ax = axs[0, 1]
for (v, s), g in H20.groupby(['variant', 'seed']):
    ax.plot(g.epoch, g.loss_rel, color='#9a9994', lw=.8, alpha=.6)
for (v, s), g in H20d.groupby(['variant', 'seed']):
    ax.plot(g.epoch, g.loss_rel, color=COL[v], lw=1.6)
m = H20.groupby('epoch').loss_rel.mean(); ax.plot(m.index, m.values, color=INK, lw=2)
ax.axvline(6, color=INK2, ls=':', lw=1)
ax.set_xlabel('Epoch'); ax.set_ylabel('Training pneumonia loss (÷ epoch-1 value)'); ax.set_xlim(.7, 11.3); ax.set_xticks(range(1, 12))
ax.set_title('b  Same runs: training loss keeps falling')
ax = axs[1, 0]
for (v, s), g in H6.groupby(['variant', 'seed']):
    ax.plot(g.epoch + np.random.default_rng(s).uniform(-.06, .06), g.rel, color=COL[v], lw=.6, alpha=.35)
m = H6.groupby('epoch').rel.agg(['mean', 'std'])
ax.errorbar(m.index, m['mean'], yerr=m['std'], color=INK, lw=2, capsize=3, label='Mean ± SD, 80 runs')
cnt = dyn[dyn.schedule == '6-epoch'].best_epoch.value_counts().sort_index()
for e, c in cnt.items(): ax.text(e, 0.004, f'{c}', ha='center', fontsize=7, color=INK)
ax.text(1, 0.004, 'best\nepoch n:', ha='center', fontsize=6.3, color=INK2)
ax.set_ylim(-.05, .012); ax.set_xlabel('Epoch'); ax.set_ylabel('Validation AUC minus run peak')
ax.set_title('c  Confirmatory 6-epoch runs (80 models)'); ax.legend(fontsize=6.5, loc='lower left')
ax = axs[1, 1]
d6 = dyn[dyn.schedule == '6-epoch'].merge(per[['variant', 'seed', 'auc']], on=['variant', 'seed'])
rng = np.random.default_rng(1)
for v in V:
    g = d6[d6.variant == v]
    ax.scatter(g.best_epoch + rng.uniform(-.18, .18, len(g)) + (V.index(v) - 1.5) * .05, g.auc, s=16, marker=MK[v], color=COL[v],
               edgecolor='white', lw=.4, label=LAB[v])
mm = d6.groupby('best_epoch').auc.mean(); ax.plot(mm.index, mm.values, color=INK, lw=1.5, marker='_', ms=14)
rho, p = N['spearman_bestepoch_vs_testauc']
ax.text(.03, .93, f'Spearman ρ = {rho:.2f}, p = {p:.3f}', transform=ax.transAxes, fontsize=7, color=INK)
ax.set_xticks([2, 3, 4]); ax.set_xlabel('Epoch of best validation AUC'); ax.set_ylabel('Test ROC-AUC')
ax.set_title('d  Later peaks did not give better models'); ax.legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=4, handletextpad=.2, columnspacing=.8)
fig.tight_layout(); save(fig, 'Figure2_training_dynamics')

# ================= Figure 3: per-seed AUC + contrasts with equivalence margins
hb = pd.read_csv(f'{O}/hier_boot.csv'); cf = pd.read_csv(f'{RESULTS_DIR}/confirmatory/contrasts_final.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.2), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]
piv = per.pivot(index='seed', columns='variant', values='auc').loc[SEEDS]
for s in SEEDS:
    ax.plot(range(4), piv.loc[s, V].values, color=GRID, lw=.7, zorder=1)
for i, v in enumerate(V):
    ax.scatter(np.full(20, i) + np.random.default_rng(i).uniform(-.08, .08, 20), piv[v], s=14, marker=MK[v], color=COL[v],
               edgecolor='white', lw=.4, zorder=3)
    mu = piv[v].mean(); se = piv[v].std() / np.sqrt(20)
    ax.errorbar(i + .28, mu, yerr=1.96 * se, fmt='o', color=INK, ms=4, capsize=3, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(['Base', 'PAAG', 'CSM', 'Full'])
ax.set_xlim(-.4, 3.9); ax.set_ylabel('Test ROC-AUC (consensus label)')
ax.set_title('a  20 seeds per variant')
ax = axs[1]
order = ['PAAG - Base', 'CSM - Base', 'Full model - Base', 'PAAG - CSM', 'Full model - PAAG', 'Full model - CSM']
names = {'PAAG - Base': 'PAAG − Base', 'CSM - Base': 'CSM − Base', 'Full model - Base': 'Full − Base', 'PAAG - CSM': 'PAAG − CSM',
         'Full model - PAAG': 'Full − PAAG', 'Full model - CSM': 'Full − CSM'}
hbm = {'PAAG - Base': 'PAAG - Base', 'CSM - Base': 'CSM - Base', 'Full model - Base': 'Full - Base', 'PAAG - CSM': 'PAAG - CSM',
       'Full model - PAAG': 'Full - PAAG', 'Full model - CSM': 'Full - CSM'}
ax.axvspan(-.01, .01, color='#f1faf6', zorder=0); ax.axvspan(-.005, .005, color='#d7f0e5', zorder=0)
ax.axvline(0, color=INK2, lw=.8)
for j, c in enumerate(order):
    r = cf[cf.contrast == c].iloc[0]; h = hb[hb.contrast == hbm[c]].iloc[0]; yy = len(order) - 1 - j
    ax.plot([h.ci90_lo, h.ci90_hi], [yy, yy], color=INK, lw=3, solid_capstyle='butt')
    ax.plot([r.ci_low, r.ci_high], [yy, yy], color=INK, lw=1)
    ax.plot(r.dAUC, yy, 'o', color='white', mec=INK, ms=5, zorder=5)
    lab = f'{r.dAUC:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]' + (f'  Holm p = {r.p_holm:.2f}' if r.primary else '')
    ax.text(.0108, yy, lab, fontsize=6.5, va='center', color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[c] for c in order][::-1])
ax.axhline(2.5, color=GRID, lw=1)
ax.set_xlim(-.0125, .0265); ax.set_xticks([-.01, 0, .01]); ax.set_xticklabels(['−0.01', '0', '+0.01'])
ax.set_xlabel('Δ test ROC-AUC (mean over 20 seeds)')
ax.set_title('b  Contrasts: thick 90% CI, thin 95% CI')
ax.text(0, -.75, '±0.005', fontsize=6.3, ha='center', color='#0f6b4b'); ax.text(-.0097, -.75, '±0.01', fontsize=6.3, color='#0f6b4b'); ax.set_ylim(-1, 5.5)
fig.tight_layout(); save(fig, 'Figure3_auc_contrasts')

# ================= Figure 4: decision churn and McNemar
mc = pd.read_csv(f'{O}/mcnemar.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8))
ax = axs[0]
cw = N['churn_within_variant_reseed']; cb = N['churn_between_same_seed']
labels = ['Base', 'PAAG', 'CSM', 'Full']; vals = [cw[v][0] * 100 for v in V]
ax.barh(np.arange(4) + 7, vals, color='#9a9994', height=.7)
bl = ['PAAG − Base', 'CSM − Base', 'Full − Base', 'PAAG − CSM', 'Full − PAAG', 'Full − CSM']
bk = ['PAAG - Base', 'CSM - Base', 'Full - Base', 'PAAG - CSM', 'Full - PAAG', 'Full - CSM']
ax.barh(np.arange(6), [cb[k] * 100 for k in bk], color=COL['base'], height=.7)
ax.set_yticks(list(range(6)) + list(np.arange(4) + 7)); ax.set_yticklabels(bl + [f'{l}: seed vs seed' for l in labels], fontsize=7)
for yv, val in zip(list(range(6)) + list(np.arange(4) + 7), [cb[k] * 100 for k in bk] + vals):
    ax.text(val + .15, yv, f'{val:.1f}%', va='center', fontsize=6.6)
ax.text(0.2, 5.75, 'Swapping modules (same seed)', fontsize=7, color=COL['base'], va='bottom', fontweight='bold')
ax.text(0.2, 10.75, 'Re-seeding (same variant)', fontsize=7, color=INK2, va='bottom', fontweight='bold')
ax.set_xlabel('Test images whose decision flips (%)'); ax.set_xlim(0, 11.5); ax.grid(axis='y', visible=False)
ax.set_title('a  Decisions that flip (spec. 0.80)')
ax = axs[1]
fr = [(mc[(mc.kind == 'module')].p < .05).mean() * 100, (mc[(mc.kind == 'reseed')].p < .05).mean() * 100]
ax.bar([0, 1], fr, color=[COL['base'], '#9a9994'], width=.55)
for i, f in enumerate(fr): ax.text(i, f + .4, f'{f:.1f}%', ha='center', fontsize=7.5)
ax.axhline(5, color=INK2, ls=':', lw=1); ax.text(1.32, 5.3, 'nominal 5%', fontsize=6.5, color=INK2, ha='right')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Module swap\n(120 pairs)', 'Re-seed\n(76 pairs)'])
ax.set_ylabel('McNemar p < 0.05 (%)'); ax.set_ylim(0, 18); ax.set_title('b  "Significant" disagreement')
fig.tight_layout(); save(fig, 'Figure4_churn')

# ================= Figure 5: ROC (seed ensembles) and calibration after Platt
d0 = load_predictions(CONF, 'base', 42); y = d0['test_y']; k = y >= 0; yk = y[k]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.2))
ax = axs[0]; ins = ax.inset_axes([.5, .1, .45, .45])
for v in V:
    ps = []; pcs = []
    for s in SEEDS:
        d = load_predictions(CONF, v, s); ps.append(d['test_p'][k])
    pe = np.mean(ps, 0)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yk, pe)
    for a_ in (ax, ins): a_.plot(fpr, tpr, color=COL[v], lw=1.3, label=f"{LAB[v]} (AUC {N['ensemble_auc'][v]:.3f})")
ax.plot([0, 1], [0, 1], color=GRID, lw=1); ins.set_xlim(0, .3); ins.set_ylim(.4, .9); ins.tick_params(labelsize=6)
ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.legend(fontsize=6.3, loc='upper left')
ax.set_title('a  ROC of 20-seed ensembles')
ax = axs[1]
for v in V:
    obs, prd = [], []
    allp, ally = [], []
    for s in SEEDS:
        d = load_predictions(CONF, v, s); sm = load_summary(CONF, v, s)
        p = d['test_p'][k]; z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        allp.append(1 / (1 + np.exp(-(sm['platt_coef'] * z + sm['platt_intercept'])))); ally.append(yk)
    allp = np.concatenate(allp); ally = np.concatenate(ally)
    qs = np.quantile(allp, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(qs, allp, side='right') - 1, 0, 9)
    ax.plot([allp[b == i].mean() for i in range(10)], [ally[b == i].mean() for i in range(10)], marker=MK[v], ms=4, color=COL[v], lw=1.2, label=LAB[v])
ax.plot([0, .45], [0, .45], color=GRID, lw=1); ax.set_xlim(0, .45); ax.set_ylim(0, .45)
ax.set_xlabel('Predicted probability (Platt-scaled on validation)'); ax.set_ylabel('Observed pneumonia frequency')
ax.legend(fontsize=6.5, loc='upper left'); ax.set_title('b  Calibration (deciles, 20 seeds pooled)')
fig.tight_layout(); save(fig, 'Figure5_roc_calibration')

# ================= Figure S1: development sweep curves (validation, seed 42, base)
names = {'80e091a5af19': '20 ep, lr 3e-4 (initial)', 'f0dc9ac04488': 'Regularised, 12 ep', 'a4cb9e818f2b': 'MIMIC-CXR init, 20 ep',
         'b029ad76da3c': 'MIMIC-CXR + regularised', 'a9642dff9011': '6 ep, patience 3 (chosen)', '6634879fc7bd': '6 ep, sqrt pos-weight',
         '4ba5b086e734': 'MIMIC-CXR, full fine-tune, 6 ep'}
pal = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948', '#008300', '#eda100']
fig, ax = plt.subplots(figsize=(7.1, 3.4))
for i, (hh, nm) in enumerate(names.items()):
    sj = load_summary(f'dev_{hh}', 'base', 42); h = pd.DataFrame(sj['history'])
    ax.plot(h.epoch, h.val_auc_ema, marker='o', ms=3, lw=1.4 if hh != 'a9642dff9011' else 2.4, color=pal[i], label=f"{nm}: {sj['best_val_auc_ema']:.4f} @ {sj['best_epoch']}")
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation ROC-AUC (EMA weights)'); ax.set_xticks(range(1, 21))
ax.legend(fontsize=6.5, loc='lower right', ncol=1); ax.set_title('Development sweep, Base variant, seed 42 (validation set only)')
fig.tight_layout(); save(fig, 'FigureS1_dev_sweep')

# ================= Figure S2: subgroup AUC
fig, ax = plt.subplots(figsize=(7.1, 2.9))
cols = [('auc_PA', 'PA view'), ('auc_AP', 'AP view'), ('auc_chexpert', 'CheXpert only'), ('auc_nih', 'ChestX-ray8 only'),
        ('auc', 'All scored'), ('auc_image_label_all', 'Original image label\n(all 12,220)'), ('auc_report_label', 'Report label')]
for j, (c, nm) in enumerate(cols):
    for i, v in enumerate(V):
        g = per[per.variant == v][c]; x = j + (i - 1.5) * .17
        ax.errorbar(x, g.mean(), yerr=g.std(), fmt=MK[v], color=COL[v], ms=4, capsize=2, label=LAB[v] if j == 0 else None)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _, n in cols], fontsize=7); ax.set_ylabel('Test ROC-AUC (mean ± SD, 20 seeds)')
ax.legend(fontsize=6.5, ncol=4, loc='lower left'); ax.set_title('Subgroups and alternative reference labels')
fig.tight_layout(); save(fig, 'FigureS2_subgroups')
print(sorted(os.listdir(F)))

# ================= Figure 6: external evaluation (PadChest)
pe = pd.read_csv(f'{H}/out_ext/per_run_external.csv'); ce = pd.read_csv(f'{H}/out_ext/contrasts_external.csv')
pi = pd.read_csv(f'{O}/per_run_full.csv')[['variant', 'seed', 'auc']].rename(columns={'auc': 'auc_int'})
mm = pe.merge(pi, on=['variant', 'seed'])
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.9), gridspec_kw={'width_ratios': [1, 1.15, 1.1]})
ax = axs[0]
for v in V:
    g = mm[mm.variant == v]
    ax.scatter(g.auc_int, g.auc, s=13, marker=MK[v], color=COL[v], edgecolor='white', lw=.4, label=LAB[v].split(' (')[0])
ax.set_xlabel('Internal test ROC-AUC'); ax.set_ylabel('PadChest ROC-AUC'); ax.legend(fontsize=6, loc='lower right', handletextpad=.2)
ax.set_title('a  Each of 80 models')
ax = axs[1]
order = ['PAAG - Base', 'CSM - Base', 'Full - Base', 'PAAG - CSM', 'Full - PAAG', 'Full - CSM']
ax.axvspan(-.01, .01, color='#f1faf6', zorder=0); ax.axvspan(-.005, .005, color='#d7f0e5', zorder=0); ax.axvline(0, color=INK2, lw=.8)
for j, c in enumerate(order):
    r = ce[ce.contrast == c].iloc[0]; yy = len(order) - 1 - j
    ax.plot([r.ci90_lo, r.ci90_hi], [yy, yy], color=INK, lw=3, solid_capstyle='butt'); ax.plot([r.ci95_lo, r.ci95_hi], [yy, yy], color=INK, lw=1)
    ax.plot(r.dAUC, yy, 'o', color='white', mec=INK, ms=4.5, zorder=5)
ax.set_yticks(range(6)); ax.set_yticklabels([c.replace(' - ', ' − ') for c in order][::-1], fontsize=7); ax.axhline(2.5, color=GRID, lw=1)
ax.set_xlim(-.014, .014); ax.set_xticks([-.01, 0, .01]); ax.set_xticklabels(['−0.01', '0', '+0.01']); ax.set_xlabel('Δ PadChest ROC-AUC')
ax.set_title('b  Paired contrasts')
ax = axs[2]
grp = [('fpr_normal', 'Normal'), ('fpr_nocom', 'No comorbidity'), ('fpr_cardiomegaly', 'Cardiomegaly'), ('fpr_pleural', 'Pleural'),
       ('fpr_heart_insufficiency', 'Heart failure')]
for i, (c, nm) in enumerate(grp):
    for j, v in enumerate(V):
        g = pe[pe.variant == v][c] * 100
        ax.bar(i + (j - 1.5) * .19, g.mean(), width=.17, color=COL[v], yerr=g.std(), error_kw={'lw': .7, 'capsize': 1.5})
ax.set_xticks(range(len(grp))); ax.set_xticklabels([n for _, n in grp], rotation=35, ha='right', fontsize=7)
ax.set_ylabel('False-positive rate, % (no pneumonia)'); ax.grid(axis='x', visible=False)
ax.set_title('c  False positives by group')
fig.tight_layout(); save(fig, 'Figure6_external_padchest')
