"""Where the analysis scripts find their inputs, and how labels are re-attached to the label-stripped predictions.

Set these environment variables (or edit the defaults) before running analysis.py / ext_analysis.py / figures.py:

  GPPM_RESULTS      results_P2 folder from the Zenodo archive (predictions.npz without labels + summary.json)
                    default: <repo>/data/results_P2
  GPPM_COHORT_CSV   cohort_from_folders.csv rebuilt with cohort/rebuild_cohort_labels.py from YOUR OWN copies of
                    the CheXpert (VisualCheXbert, CheXbert) and NIH ChestX-ray14 label files
                    default: <repo>/data/cohort_from_folders.csv
  GPPM_PADCHEST_MANIFEST  padchest_external_manifest.csv rebuilt with cohort/padchest_filter.py from YOUR OWN
                    PadChest label CSV (BIMCV licence)        default: <repo>/data/padchest_external_manifest.csv
  GPPM_EXTERNAL_NPZ external_padchest_all.npz from the Zenodo archive
                    default: <repo>/data/results_P2/a9642dff9011/external_padchest_all.npz

Labels are not redistributed (dataset licences). The published prediction files keep the image order used in
training, so labels are re-attached by POSITION from the rebuilt cohort and checked file name by file name.
"""
import os
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get('GPPM_RESULTS', os.path.join(REPO, 'data', 'results_P2'))
COHORT_CSV = os.environ.get('GPPM_COHORT_CSV', os.path.join(REPO, 'data', 'cohort_from_folders.csv'))
PADCHEST_MANIFEST = os.environ.get('GPPM_PADCHEST_MANIFEST', os.path.join(REPO, 'data', 'padchest_external_manifest.csv'))
EXTERNAL_NPZ = os.environ.get('GPPM_EXTERNAL_NPZ', os.path.join(R, 'a9642dff9011', 'external_padchest_all.npz'))
CONFIG_DIR = os.path.join(REPO, 'config')
RESULTS_DIR = os.path.join(REPO, 'results')
SEED_REGISTRY = os.path.join(CONFIG_DIR, 'seed_registry.json')

_COHORT = None


def cohort():
    global _COHORT
    if _COHORT is None:
        if not os.path.exists(COHORT_CSV):
            raise FileNotFoundError(f"{COHORT_CSV} not found. Rebuild it with cohort/rebuild_cohort_labels.py "
                                    "(see README, 'Reproducing the analysis').")
        _COHORT = pd.read_csv(COHORT_CSV, low_memory=False)
    return _COHORT


def _rows(split):
    d = cohort()
    d = d[d['split'] == split].reset_index(drop=True)
    idx = d['fname'] if 'fname' in d.columns else d['image_id'] + d['ext']
    return d, idx.astype(str).values


def attach_labels(d, folder_labels=False):
    """Add val_y / test_y (and the image, report and consensus labels) to a label-stripped predictions dict.

    folder_labels=True reproduces the 20-epoch development runs (b6253a0a3fe1), whose reference was the folder
    a file sat in; those files were written in folder-walk order (group folder, then file name)."""
    if 'test_y' in d and 'val_y' in d:
        return d                                                  # original, unstripped file
    for split, key in [('Validation', 'val'), ('Test', 'test')]:
        if f'{key}_index' not in d:
            continue
        rows, idx = _rows(split)
        if folder_labels:
            order = np.lexsort((rows['fname'].values, rows['folder_group'].values))
            rows, idx = rows.iloc[order].reset_index(drop=True), idx[order]
            y = (rows['folder_group'] == 'Pneumonia_Infectious').astype(int).values
        else:
            y = rows['y_consensus'].astype(int).values
        if not np.array_equal(idx, np.asarray(d[f'{key}_index']).astype(str)):
            raise ValueError(f"{key}: image order in the rebuilt cohort differs from the predictions file. "
                             "Rebuild the cohort with cohort/rebuild_cohort_labels.py and cohort/split_assignment.csv.")
        d[f'{key}_y'] = y
        if key == 'test' and not folder_labels:
            for e in ('y_image', 'y_report', 'y_consensus'):
                d[f'test_{e}'] = rows[e].astype(int).values
    return d


def load_predictions(cfg, variant, seed, folder_labels=False):
    d = dict(np.load(os.path.join(R, cfg, variant, f'seed_{seed}', 'predictions.npz')))
    return attach_labels(d, folder_labels=folder_labels)


def load_summary(cfg, variant, seed):
    """summary.json is small and is also in the repository (results/runs)."""
    for root in (R, os.path.join(RESULTS_DIR, 'runs')):
        f = os.path.join(root, cfg, variant, f'seed_{seed}', 'summary.json')
        if os.path.exists(f):
            return json.load(open(f))
    raise FileNotFoundError(f'summary.json for {cfg}/{variant}/seed_{seed}')
