"""Select the PadChest external test cohort and map its physician labels to the study's targets.
Usage: python padchest_filter.py PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz padchest_external_manifest.csv"""
import sys, ast
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False)
d['age'] = d['StudyDate_DICOM'] // 10000 - d['PatientBirth']
d['lab'] = d['Labels'].map(lambda x: sorted({s.strip().lower() for s in ast.literal_eval(x) if isinstance(s, str) and s.strip()})
                           if isinstance(x, str) and x.startswith('[') else None)
f = d[(d['MethodLabel'] == 'Physician') & d['Projection'].isin(['PA', 'AP', 'AP_horizontal']) & (d['age'] >= 65)
      & (d['Pediatric'] == 'No') & d['lab'].notna()].copy()

has = lambda terms: f['lab'].map(lambda l: int(any(t in l for t in terms)))
# primary reference: physician-assigned pneumonia diagnosis
f['pneumonia'] = has(['pneumonia', 'atypical pneumonia'])
# comorbidity groups, mapped to the five heads the models were trained with
f['com_Cardiomegaly'] = has(['cardiomegaly'])
f['com_Pleural_Extrapulmonary_Volume_Disease'] = has(['pleural effusion', 'costophrenic angle blunting', 'atelectasis',
    'laminar atelectasis', 'lobar atelectasis', 'segmental atelectasis', 'round atelectasis', 'atelectasis basal',
    'total atelectasis', 'pneumothorax', 'pleural thickening', 'apical pleural thickening', 'calcified pleural thickening',
    'hiatal hernia'])
f['com_Weakly_Supervised_Uncertain'] = has(['pulmonary edema'])               # non-infectious airspace opacity (edema)
f['com_Focal_Lesion_Chronic_Change'] = has(['emphysema', 'pulmonary fibrosis', 'nodule', 'multiple nodules',
                                             'pulmonary mass', 'mass'])
f['com_No_Finding'] = f['lab'].map(lambda l: int(l == ['normal']))
f['heart_insufficiency'] = has(['heart insufficiency'])
f['pleural_effusion'] = has(['pleural effusion'])
# airspace findings without a pneumonia diagnosis: possible unlabelled infection (excluded in a sensitivity analysis)
f['airspace_no_pneumonia'] = ((has(['consolidation', 'infiltrates', 'alveolar pattern', 'ground glass pattern']) == 1)
                              & (f['pneumonia'] == 0)).astype(int)
f['view'] = np.where(f['Projection'] == 'PA', 'PA', 'AP')
cols = ['ImageID', 'ImageDir', 'PatientID', 'StudyID', 'StudyDate_DICOM', 'age', 'PatientSex_DICOM', 'view', 'Projection',
        'Modality_DICOM', 'Manufacturer_DICOM', 'BitsStored_DICOM', 'pneumonia', 'com_Cardiomegaly',
        'com_Focal_Lesion_Chronic_Change', 'com_No_Finding', 'com_Pleural_Extrapulmonary_Volume_Disease',
        'com_Weakly_Supervised_Uncertain', 'heart_insufficiency', 'pleural_effusion', 'airspace_no_pneumonia']
m = f[cols].copy(); m['labels'] = f['lab'].map(lambda l: '|'.join(l))
m.to_csv(out, index=False)
print(f"{len(m)} images, {m.PatientID.nunique()} patients, {m.pneumonia.sum()} pneumonia "
      f"({100 * m.pneumonia.mean():.1f}%), age median {m.age.median():.0f}; archives needed: {sorted(m.ImageDir.unique())}")
for c in [c for c in m.columns if c.startswith('com_')] + ['heart_insufficiency', 'pleural_effusion', 'airspace_no_pneumonia']:
    print(f"  {c:45s} {m[c].sum():6d}   with pneumonia: {int(((m[c] == 1) & (m.pneumonia == 1)).sum())}")
print(m.groupby('ImageDir').agg(images=('pneumonia', 'size'), pneumonia=('pneumonia', 'sum')).T.to_string())
