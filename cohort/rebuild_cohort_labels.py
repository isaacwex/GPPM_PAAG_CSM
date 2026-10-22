"""Rebuild the study cohort with labels (cohort_from_folders.csv) from the published image list and YOUR OWN copies
of the official label files. Labels are not redistributed in this repository (CheXpert and NIH ChestX-ray14 terms).

    python cohort/rebuild_cohort_labels.py \
        --split cohort/split_assignment.csv \
        --visual  train_visualCheXbert.csv      # CheXpert VisualCheXbert labels (Stanford AIMI)
        --report  train_cheXbert.csv            # CheXpert report labels (CheXbert)
        --nih     Data_Entry_2017_v2020.csv     # NIH ChestX-ray14 metadata
        --out     data/cohort_from_folders.csv

The logic is the notebook's P1b cell (src/protocol2/p1b_label_folders.py), with the folder walk replaced by the
published image list, so row order and labels are identical to those used in training. The CheXpert file names in
the study carry no study ID, so one file can stand for several studies of one patient; a label is kept when all
candidate studies agree and set to -1 (masked) when they disagree.
"""
import argparse
import re
import numpy as np
import pandas as pd

CXP_OBS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
           'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']
GROUPS = {'Pneumonia_Infectious': 'Pneumonia (Infectious)', 'Cardiomegaly': 'Cardiomegaly',
          'Pleural_Extrapulmonary_Volume_Disease': 'Pleural/Extrapulmonary/Volume Disease',
          'Weakly_Supervised_Uncertain': 'Airspace Opacity (non-infectious)',
          'Focal_Lesion_Chronic_Change': 'Focal Lesion/Chronic Change', 'No_Finding': 'No Finding'}
COMORB = ['Cardiomegaly', 'Pleural/Extrapulmonary/Volume Disease', 'Airspace Opacity (non-infectious)',
          'Focal Lesion/Chronic Change', 'No Finding']
NIH_GRP = {'Cardiomegaly': 'Cardiomegaly', 'Atelectasis': 'Pleural/Extrapulmonary/Volume Disease',
           'Effusion': 'Pleural/Extrapulmonary/Volume Disease', 'Hernia': 'Pleural/Extrapulmonary/Volume Disease',
           'Pleural_Thickening': 'Pleural/Extrapulmonary/Volume Disease',
           'Pneumothorax': 'Pleural/Extrapulmonary/Volume Disease', 'Consolidation': 'Airspace Opacity (non-infectious)',
           'Edema': 'Airspace Opacity (non-infectious)', 'Infiltration': 'Airspace Opacity (non-infectious)',
           'Emphysema': 'Focal Lesion/Chronic Change', 'Fibrosis': 'Focal Lesion/Chronic Change',
           'Mass': 'Focal Lesion/Chronic Change', 'Nodule': 'Focal Lesion/Chronic Change'}
NIH_OBS = {'Cardiomegaly': 'Cardiomegaly', 'Edema': 'Edema', 'Consolidation': 'Consolidation',
           'Atelectasis': 'Atelectasis', 'Pneumothorax': 'Pneumothorax', 'Effusion': 'Pleural Effusion',
           'Mass': 'Lung Lesion', 'Nodule': 'Lung Lesion', 'No Finding': 'No Finding'}


def cxp_rule_group(v):
    g = np.full(len(v), None, dtype=object)
    pleural = v[['Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other']].eq(1).any(axis=1).values
    for cond, name in [(v['No Finding'].eq(1).values, 'No_Finding'), (v['Edema'].eq(1).values, 'Weakly_Supervised_Uncertain'),
                       (pleural, 'Pleural_Extrapulmonary_Volume_Disease'), (v['Cardiomegaly'].eq(1).values, 'Cardiomegaly'),
                       (v['Pneumonia'].eq(1).values, 'Pneumonia_Infectious')]:      # reverse priority: last wins
        g[cond] = name
    return g


def rebuild(split_csv, visual_csv, report_csv, nih_csv):
    files = pd.read_csv(split_csv, low_memory=False)
    meta_cols = ['image_id', 'source', 'patient_id', 'view', 'sex', 'age']
    meta = files[meta_cols].copy()
    files = files[['split', 'folder_group', 'fname', 'file_rel', 'image_id']]
    # ---- CheXpert candidates
    vis = pd.read_csv(visual_csv)
    rep = pd.read_csv(report_csv)[['Path', 'Pneumonia']].rename(columns={'Pneumonia': 'rep_pneu'})
    v = vis[(vis['Frontal/Lateral'] == 'Frontal') & (vis['Age'] >= 65)].merge(rep, on='Path', how='left')
    v['pid'] = v['Path'].str.extract(r'(patient\d+)', expand=False)
    v['viewfile'] = v['Path'].str.extract(r'(view\d+_frontal)', expand=False)
    v['folder_group'] = cxp_rule_group(v)
    v = v.dropna(subset=['folder_group'])
    v['fname'] = ('CheXpert_' + v['pid'] + '_' + v['Age'].astype(int).astype(str) + '_' + v['Sex'] + '_' +
                  v['AP/PA'].astype(str) + '_' + v['viewfile'] + '.jpg')
    v['y_image'] = v['Pneumonia'].eq(1).astype(int)
    rp = v['rep_pneu']
    v['y_report'] = np.select([rp.eq(1), rp.eq(-1), rp.eq(0)], [1, -1, 0], 9)
    rpos = rp.isin([1.0, -1.0])
    v['y_consensus'] = np.where(v['y_image'].eq(1) & rpos, 1, np.where(v['y_image'].eq(0) & ~rpos, 0, -1))
    v['com::Cardiomegaly'] = v['Cardiomegaly'].eq(1).astype(int)
    v['com::Pleural/Extrapulmonary/Volume Disease'] = v[['Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                                                          'Pleural Other']].eq(1).any(axis=1).astype(int)
    v['com::Airspace Opacity (non-infectious)'] = v['Edema'].eq(1).astype(int)
    v['com::No Finding'] = v['No Finding'].eq(1).astype(int)
    for o in CXP_OBS:
        v['obs::' + o] = v[o].eq(1).astype(int)
    lab_cols = ['y_image', 'y_report', 'y_consensus'] + [c for c in v.columns if c.startswith(('com::', 'obs::'))]
    g = v.groupby(['fname', 'folder_group'])
    first, nuniq = g[lab_cols].first(), g[lab_cols].nunique()
    agg = first.where(nuniq.eq(1), -1)
    agg['y_report'] = first['y_report'].where(nuniq['y_report'].eq(1), -2)
    agg['n_candidates'] = g.size()
    agg = agg.reset_index()
    agg['com::Focal Lesion/Chronic Change'] = -1
    cx = files[files['fname'].str.startswith('CheXpert')].merge(agg, on=['fname', 'folder_group'], how='left')
    # ---- NIH (exact: the file name carries the image index)
    de = pd.read_csv(nih_csv)
    lab = dict(zip(de['Image Index'], de['Finding Labels'].astype(str)))
    nx = files[files['fname'].str.startswith('ChestX-ray8')].copy()
    nrows = []
    for f in nx['fname']:
        m = re.search(r'_(\d{8}_\d{3})\.png$', f)
        fs = set(lab.get(m.group(1) + '.png', 'NA').split('|')) if m else {'NA'}
        r = {'y_image': int('Pneumonia' in fs), 'y_report': int('Pneumonia' in fs), 'y_consensus': int('Pneumonia' in fs),
             'n_candidates': 1 if m and m.group(1) + '.png' in lab else 0}
        for gname in COMORB:
            r['com::' + gname] = int(fs == {'No Finding'}) if gname == 'No Finding' else int(any(NIH_GRP.get(z) == gname for z in fs))
        for o in CXP_OBS:
            r['obs::' + o] = int(any(NIH_OBS.get(z) == o for z in fs)) if o in set(NIH_OBS.values()) else -1
        nrows.append(r)
    nx = pd.concat([nx.reset_index(drop=True), pd.DataFrame(nrows)], axis=1)
    coh = pd.concat([cx, nx], ignore_index=True).merge(meta, on='image_id', how='left', validate='one_to_one')
    coh['group'] = coh['folder_group'].map(GROUPS)
    coh['ext'] = '.' + coh['fname'].str.rsplit('.', n=1).str[1]
    for c in [c for c in coh.columns if c.startswith(('y_', 'com::', 'obs::'))]:
        coh[c] = coh[c].fillna(-1).astype(int)
    coh['y_image'] = (coh['folder_group'] == 'Pneumonia_Infectious').astype(int)   # the folder defined y_image
    assert coh['image_id'].is_unique
    return coh


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='cohort/split_assignment.csv')
    ap.add_argument('--visual', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--nih', required=True)
    ap.add_argument('--out', default='data/cohort_from_folders.csv')
    a = ap.parse_args()
    coh = rebuild(a.split, a.visual, a.report, a.nih)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    coh.to_csv(a.out, index=False)
    print(f"{len(coh)} images written to {a.out}\n" + pd.crosstab(coh['y_consensus'], coh['split']).to_string())
