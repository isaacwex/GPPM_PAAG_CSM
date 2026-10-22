"""
cohort_builder_v2.py  —  rebuild the geriatric cohort with collision-free filenames and full labels.

WHY: the original export named CheXpert images  CheXpert_<patient>_<age>_<sex>_<view>_<viewN>_frontal.jpg.
That name has no study ID, so different studies of the same patient at the same age overwrite each other:
85,265 eligible frontal images map to only 41,349 names (51% lost), and the current manifest has 41,765
CheXpert rows for 22,801 names. This script keeps your PATIENT-LEVEL SPLIT and your six-group priority rule
(validated: it reproduces every (filename, group) pair in subset_dataset_manifest.csv), but
  * names every image uniquely (study ID included) and, optionally, copies it from the ORIGINAL datasets;
  * attaches labels from BOTH CheXpert labelers (VisualCheXbert image labels, CheXbert report labels) and
    NIH Data_Entry_2017, so comorbidities are true multi-labels (pneumonia + cardiomegaly can co-exist);
  * adds pneumonia targets: y_image (old definition), y_report (report-positive), y_consensus
    (1 = both labelers positive/uncertain-report, 0 = both negative, -1 = labelers disagree -> masked).

Run it where the original images are (your E: drive, or Paperspace after uploading the originals):
    python cohort_builder_v2.py --manifest subset_dataset_manifest.csv --visual train_visualCheXbert.csv \
        --report train_cheXbert.csv --nih Data_Entry_2017_v2020.csv --out /notebooks/cohort_v2 \
        --chexpert-root "E:/CheXpert-v1.0-small/.."  --nih-root "E:/NIH/images"   [--copy]
Without --copy it only writes cohort_v2_manifest.csv (+ report) so you can inspect it first.
"""
import argparse, os, re, shutil, json
import numpy as np, pandas as pd

CXP_OBS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
           'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices',
           'No Finding']                                    # 13 auxiliary observations (Pneumonia is the target)
GROUPS = ['Pneumonia (Infectious)', 'Cardiomegaly', 'Pleural/Extrapulmonary/Volume Disease',
          'Airspace Opacity (non-infectious)', 'Focal Lesion/Chronic Change', 'No Finding']   # priority order
COMORB = ['Cardiomegaly', 'Pleural/Extrapulmonary/Volume Disease', 'Airspace Opacity (non-infectious)',
          'Focal Lesion/Chronic Change', 'No Finding']
NIH_GROUP = {'Cardiomegaly': 'Cardiomegaly', 'Atelectasis': 'Pleural/Extrapulmonary/Volume Disease',
             'Effusion': 'Pleural/Extrapulmonary/Volume Disease', 'Hernia': 'Pleural/Extrapulmonary/Volume Disease',
             'Pleural_Thickening': 'Pleural/Extrapulmonary/Volume Disease',
             'Pneumothorax': 'Pleural/Extrapulmonary/Volume Disease',
             'Consolidation': 'Airspace Opacity (non-infectious)', 'Edema': 'Airspace Opacity (non-infectious)',
             'Infiltration': 'Airspace Opacity (non-infectious)', 'Emphysema': 'Focal Lesion/Chronic Change',
             'Fibrosis': 'Focal Lesion/Chronic Change', 'Mass': 'Focal Lesion/Chronic Change',
             'Nodule': 'Focal Lesion/Chronic Change', 'No Finding': 'No Finding'}
NIH_TO_OBS = {'Cardiomegaly': 'Cardiomegaly', 'Edema': 'Edema', 'Consolidation': 'Consolidation',
              'Atelectasis': 'Atelectasis', 'Pneumothorax': 'Pneumothorax', 'Effusion': 'Pleural Effusion',
              'Mass': 'Lung Lesion', 'Nodule': 'Lung Lesion', 'No Finding': 'No Finding'}
# NIH has no label for the other CheXpert observations -> unknown (-1) for NIH images.
NIH_KNOWN_OBS = set(NIH_TO_OBS.values())


def cxp_group(r):
    if r['Pneumonia'] == 1: return GROUPS[0]
    if r['Cardiomegaly'] == 1: return GROUPS[1]
    if any(r[c] == 1 for c in ['Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other']): return GROUPS[2]
    if r['Edema'] == 1: return GROUPS[3]
    if r['No Finding'] == 1: return GROUPS[5]
    return None


def cxp_comorb(r):
    return {'Cardiomegaly': int(r['Cardiomegaly'] == 1),
            'Pleural/Extrapulmonary/Volume Disease': int(any(r[c] == 1 for c in ['Atelectasis', 'Pneumothorax',
                                                                                    'Pleural Effusion', 'Pleural Other'])),
            'Airspace Opacity (non-infectious)': int(r['Edema'] == 1),
            'Focal Lesion/Chronic Change': -1,              # no CheXpert label maps to this group
            'No Finding': int(r['No Finding'] == 1)}


def nih_group(findings):
    if 'Pneumonia' in findings: return GROUPS[0]
    for g in GROUPS[1:]:
        if any(NIH_GROUP.get(f) == g for f in findings): return g
    return None


def build(a):
    man = pd.read_csv(a.manifest)
    man['split'] = man['Image Path'].str.extract(r'[\\/](Train|Validation|Test)[\\/]', expand=False)
    split_of = {}
    for pid, s in zip(man['Patient ID'].astype(str), man['split']):
        split_of.setdefault(pid, set()).add(s)
    leak = {p: s for p, s in split_of.items() if len(s) > 1}
    assert not leak, f"patients in >1 split in the manifest: {list(leak)[:5]}"
    split_of = {p: next(iter(s)) for p, s in split_of.items()}

    # ---------------------------------------------------------------- CheXpert
    vis = pd.read_csv(a.visual)
    rep = pd.read_csv(a.report).set_index('Path')
    v = vis[(vis['Frontal/Lateral'] == 'Frontal') & (vis['Age'] >= 65)].copy()
    v['pid'] = v['Path'].str.extract(r'(patient\d+)', expand=False)
    v = v[v['pid'].isin(split_of)]
    v['study'] = v['Path'].str.extract(r'(study\d+)', expand=False)
    v['viewfile'] = v['Path'].str.extract(r'(view\d+_frontal)', expand=False)
    v['group'] = v.apply(cxp_group, axis=1)
    v = v.dropna(subset=['group'])
    r = rep.reindex(v['Path'])
    rp = r['Pneumonia'].values
    yi = (v['Pneumonia'] == 1).astype(int).values
    yr_pos = np.isin(rp, [1.0, -1.0])
    rows = []
    for k, (_, x) in enumerate(v.iterrows()):
        cons = 1 if (yi[k] == 1 and yr_pos[k]) else (0 if (yi[k] == 0 and not yr_pos[k]) else -1)
        row = {'image_id': f"CheXpert_{x['pid']}_{x['study']}_{int(x['Age'])}_{x['Sex']}_{x['AP/PA']}_{x['viewfile']}",
               'source': 'CheXpert', 'patient_id': x['pid'], 'split': split_of[x['pid']],
               'age': int(x['Age']), 'sex': x['Sex'], 'view': x['AP/PA'], 'group': x['group'],
               'y_image': int(yi[k]), 'y_report': {1.0: 1, -1.0: -1, 0.0: 0}.get(rp[k], 9),  # 9 = not mentioned
               'y_consensus': cons, 'src_rel': x['Path'], 'ext': '.jpg'}
        for g, val in cxp_comorb(x).items():
            row['com::' + g] = val
        for o in CXP_OBS:
            row['obs::' + o] = int(x[o] == 1)
        rows.append(row)
    cxp = pd.DataFrame(rows)

    # ---------------------------------------------------------------- NIH ChestX-ray8
    de = pd.read_csv(a.nih)
    de['pid'] = de['Patient ID'].astype(str)
    d = de[(de['Patient Age'] >= 65) & de['View Position'].isin(['AP', 'PA']) & de['pid'].isin(split_of)].copy()
    rows = []
    for _, x in d.iterrows():
        f = set(str(x['Finding Labels']).split('|'))
        g = nih_group(f)
        if g is None:
            continue
        stem = os.path.splitext(x['Image Index'])[0]
        row = {'image_id': f"ChestX-ray8_{x['pid']}_{int(x['Patient Age'])}_{x['Patient Sex']}_{x['View Position']}_{stem}",
               'source': 'ChestX-ray8', 'patient_id': x['pid'], 'split': split_of[x['pid']],
               'age': int(x['Patient Age']), 'sex': x['Patient Sex'], 'view': x['View Position'], 'group': g,
               'y_image': int('Pneumonia' in f), 'y_report': int('Pneumonia' in f),
               'y_consensus': int('Pneumonia' in f),         # single (report-mined) labeler for NIH
               'src_rel': x['Image Index'], 'ext': '.png'}
        for gg in COMORB:
            row['com::' + gg] = int(f == {'No Finding'}) if gg == 'No Finding' else int(any(NIH_GROUP.get(z) == gg for z in f))
        for o in CXP_OBS:
            row['obs::' + o] = int(any(NIH_TO_OBS.get(z) == o for z in f)) if o in NIH_KNOWN_OBS else -1
        rows.append(row)
    nih = pd.DataFrame(rows)

    coh = pd.concat([cxp, nih], ignore_index=True)
    assert coh['image_id'].is_unique, "image_id collision"
    if not a.all_patient_images:
        # Keep exactly the images your manifest selected. A CheXpert manifest row is identified by its old
        # (collision-prone) filename + group; every study that produced that filename is kept, each now unique.
        fn = man['Image Path'].str.split(r'[\\/]').str[-1]
        keys = set(zip(fn, man['Group Diagnosis']))
        old = np.where(coh['source'] == 'CheXpert',
                       coh['image_id'].str.replace(r'_study\d+', '', regex=True) + '.jpg',
                       'ChestX-ray8_' + coh['patient_id'] + '_' + coh['age'].astype(str) + '_' + coh['sex'] + '_' +
                       coh['view'] + '_' + coh['src_rel'])
        keep = pd.Series([(o_, g_) in keys for o_, g_ in zip(old, coh['group'])], index=coh.index)
        nih_keep = coh['source'].eq('ChestX-ray8') & coh['src_rel'].isin(set(fn.str.extract(r'_(\d{8}_\d{3}\.png)$')[0]))
        keep = keep | nih_keep
        print(f"--only-manifest-images: keeping {int(keep.sum())} of {len(coh)} images "
              f"(use --all-patient-images to keep every eligible image of the manifest's patients)")
        coh = coh[keep].reset_index(drop=True)
    coh['file_rel'] = coh['split'] + '/' + coh['image_id'] + coh['ext']
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- report
    rep_lines = [f"images: {len(coh)} ({(coh.source == 'CheXpert').sum()} CheXpert, {(coh.source == 'ChestX-ray8').sum()} ChestX-ray8)",
                 f"patients: {coh.patient_id.nunique()} | patients in >1 split: "
                 f"{int(coh.groupby('patient_id')['split'].nunique().gt(1).sum())}",
                 "\nimages by split x group:\n" + pd.crosstab(coh['group'], coh['split'], margins=True).to_string(),
                 "\npneumonia targets by split (counts of 1 / 0 / -1 masked):"]
    for t in ['y_image', 'y_consensus']:
        rep_lines.append(f"  {t}:\n" + pd.crosstab(coh[t], coh['split']).to_string())
    rep_lines.append("\ntrue co-occurrence among y_consensus == 1 (share with each comorbidity group):\n" +
                     coh.loc[coh.y_consensus == 1, [c for c in coh if c.startswith('com::')]].replace(-1, np.nan)
                     .mean().round(3).to_string())
    txt = '\n'.join(rep_lines)
    print(txt)
    open(os.path.join(a.out, 'cohort_v2_report.txt'), 'w').write(txt)
    coh.to_csv(os.path.join(a.out, 'cohort_v2_manifest.csv'), index=False)
    print(f"\nwrote {os.path.join(a.out, 'cohort_v2_manifest.csv')}")

    # ---------------------------------------------------------------- optional copy from the original datasets
    if a.copy:
        nih_index = {}
        if a.nih_root:
            for root, _, files in os.walk(a.nih_root):
                for fn in files:
                    if fn.endswith('.png'):
                        nih_index[fn] = os.path.join(root, fn)
        missing = []
        for row in coh.itertuples(index=False):
            dst = os.path.join(a.out, 'images', row.file_rel)
            if os.path.exists(dst):
                continue
            if row.source == 'CheXpert':
                rel = row.src_rel.split('CheXpert-v1.0', 1)[-1].lstrip('/\\')
                cands = [os.path.join(a.chexpert_root, 'CheXpert-v1.0', rel), os.path.join(a.chexpert_root, rel),
                         os.path.join(a.chexpert_root, row.src_rel)]
                src = next((c for c in cands if os.path.exists(c)), None)
            else:
                src = nih_index.get(row.src_rel)
            if src is None:
                missing.append(row.src_rel); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"copy done; missing source files: {len(missing)}")
        if missing:
            pd.Series(missing).to_csv(os.path.join(a.out, 'missing_sources.csv'), index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--visual', required=True)
    p.add_argument('--report', required=True); p.add_argument('--nih', required=True)
    p.add_argument('--out', required=True); p.add_argument('--chexpert-root', default='')
    p.add_argument('--nih-root', default=''); p.add_argument('--copy', action='store_true')
    p.add_argument('--all-patient-images', action='store_true')
    build(p.parse_args())
