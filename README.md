# GPPM_PAAG_CSM

Code, configuration and results for:

> **Attention gating and comorbidity suppression do not improve pneumonia detection on geriatric chest radiographs: a pre-specified 20-seed ablation of DenseNet-121 with an analysis of training length.**
> Isaac Wafula Wekesa, Jael Sanyanda Wekesa, Judy Muthoni Gateri, Philemon Onyango Opondo. Jomo Kenyatta University of Agriculture and Technology. Submitted to *PeerJ Computer Science*.

Four DenseNet-121 variants (Base, +PAAG spatial attention gate, +CSM comorbidity suppression, Full = both) were trained with 20 pre-registered seeds each (80 models) under one frozen configuration (`a9642dff9011`), on frontal radiographs of patients aged 65 and over from CheXpert and NIH ChestX-ray14, and evaluated on a held-out test set and on an external PadChest cohort.

| Variant | Internal test AUC (mean ± SD, 20 seeds) | PadChest AUC (mean ± SD) |
|---|---|---|
| Base | 0.8456 ± 0.0032 | 0.7539 ± 0.0099 |
| PAAG | 0.8446 ± 0.0057 | 0.7526 ± 0.0082 |
| CSM | 0.8450 ± 0.0045 | 0.7540 ± 0.0089 |
| Full | 0.8448 ± 0.0041 | 0.7506 ± 0.0094 |

All module-versus-Base differences lie within ±0.01 AUC (equivalence tests in the paper). Every number in the manuscript is produced by the scripts in `analysis/`; published copies of their outputs are in `results/manuscript_numbers/`.

## What is where

| Folder | Contents |
|---|---|
| `notebooks/` | `GPPM_PAAG_CSM_v10_Protocol2.ipynb`: the notebook that trained and evaluated all models reported (cells P0–P10). `GPPM_PAAG_CSM_v4_MS_5_ext_Protocol1.ipynb`: the exploratory Protocol 1 notebook (4 variants × 5 seeds) that produced the Protocol 1 results in `results/protocol1_five_seeds/`. Outputs cleared. |
| `cohort/` | `split_assignment.csv` (every image: ID, file name, split, folder group, source, patient, view, sex, age; **no labels**), `rebuild_cohort_labels.py` (rebuilds labels from your own copies of the official label files), `cohort_builder_v2.py`, `padchest_filter.py` (selects the external cohort), `padchest_image_ids.csv` (the 12,038 PadChest images used). |
| `config/` | `seed_registry.json` (40 seeds drawn before training; first 20 used), `frozen_config_a9642dff9011.json` (configuration and software versions), `dev_sweep.csv` (the development sweep that fixed the schedule). |
| `results/confirmatory/` | Per-run metrics of the 80 models and the pre-specified contrasts (interim and final). |
| `results/runs/<config>/<variant>/seed_<s>/` | `summary.json` (per-epoch training history, best epoch, Platt parameters) and `run_manifest.json` for all 109 runs: 80 confirmatory, 16 twenty-epoch development runs, 13 sweep runs. |
| `results/localisation/` | Grad-CAM and attention mass in lung and heart masks (first run with the central-box fallback, and the anatomical run with ChestX-Det PSPNet masks). |
| `results/external_padchest/` | PadChest per-run results, contrasts and secondary analyses. |
| `results/protocol1_five_seeds/` | Summary results of the exploratory Protocol 1 study (4 variants × 5 seeds) quoted in the Introduction and Discussion: per-seed metrics, cross-seed synthesis, DeLong/McNemar tests, source- and view-stratified results, localisation IoU means. Per-image files and plots are omitted. |
| `results/label_reattachment_check.csv` | For every run, confirmation that labels rebuilt from `cohort/split_assignment.csv` equal the labels used in training. |
| `results/manuscript_numbers/` | Every table-level output of `analysis/` (`numbers.json`, bootstrap, DeLong, McNemar, training dynamics …). |
| `analysis/` | `analysis.py` (internal), `ext_analysis.py` (PadChest), `figures.py`, `paths.py` (input locations and label re-attachment). |
| `figures/` | Figures 1–6 and S1–S3 as submitted. |

Stored on **Zenodo**, record [10.5281/zenodo.23590668](https://doi.org/10.5281/zenodo.23590668) (too large for GitHub, or derived from the training data):
model weights of the 80 confirmatory models (`best_ema.pth`, in 16 zips of 5 seeds: `GPPM_PAAG_CSM_checkpoints_<variant>_part<k>of4.zip`), label-free validation and test predictions of every run, and the PadChest scores of the 80 models (`external_padchest_all.npz`). DOI: see *Citation* below.

## Data and licences

Images and labels are **not** redistributed. Obtain them from the providers under their terms:
CheXpert and the VisualCheXbert/CheXbert labels (Stanford AIMI), NIH ChestX-ray14 (`Data_Entry_2017_v2020.csv`), and PadChest (BIMCV; apply through BIMCV's own form). Never commit a private download link to this repository: keep `PADCHEST_SHARE = ''` in the notebook.

## Reproducing the analysis (no GPU needed)

```bash
pip install -r requirements.txt
# 1. Weights/predictions from Zenodo -> data/results_P2/...
unzip GPPM_PAAG_CSM_predictions.zip -d data/            # creates data/predictions/results_P2
export GPPM_RESULTS=data/predictions/results_P2
# 2. Labels, from your own copies of the official files
python cohort/rebuild_cohort_labels.py --visual train_visualCheXbert.csv --report train_cheXbert.csv \
       --nih Data_Entry_2017_v2020.csv --out data/cohort_from_folders.csv
python cohort/padchest_filter.py PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz data/padchest_external_manifest.csv
# 3. Every number and figure
python analysis/analysis.py && python analysis/ext_analysis.py && python analysis/figures.py
```

`analysis/paths.py` re-attaches the labels to each prediction file by position and checks every file name, so a wrong or differently ordered cohort stops with an error rather than producing different numbers. `results/label_reattachment_check.csv` records that this round trip reproduced the training labels exactly for all 80 confirmatory runs. (The 16 twenty-epoch development runs used folder labels; their published results are in `results/manuscript_numbers/internal/old_20ep_folder.csv` and `analysis.py` falls back to that file.)

## Retraining

Open `notebooks/GPPM_PAAG_CSM_v10_Protocol2.ipynb` on a CUDA machine (we used an NVIDIA RTX A6000, about 3.6 min per epoch), set `BASE_DIR`, put the label CSVs in `BASE_DIR/CSVs`, and run the cells in order. P1 prints the configuration hash; it must read `a9642dff9011`. Seeds are trained in registry order and the analysis (P8) refuses to report until all 20 confirmatory seeds of all four variants are complete.

## Citation

See `CITATION.cff`.

- Weights and label-free predictions (Protocol 2): https://doi.org/10.5281/zenodo.23590668
- Code and results (this repository, archived at each release): https://doi.org/10.5281/zenodo.22945068
- Weights of the exploratory Protocol 1 study (5 seeds): https://doi.org/10.5281/zenodo.23598061

## Licence

Code: MIT (`LICENSE`). Results, figures and Zenodo files: CC BY 4.0. The models are research artefacts and are not for clinical use.
