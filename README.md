# GPPM: PAAG/CSM Geriatric Pneumonia Ablation

Code, data-split manifests, and results for a factorial ablation study of two architectural additions to
a DenseNet-121 chest-radiograph classifier — a Pathology-Aware Attention Gate (PAAG) and a Comorbidity
Suppressor Module (CSM) — evaluated on a geriatric (ages 65–106) cohort pooled from CheXpert and
NIH ChestX-ray8/ChestX-ray14.

The accompanying manuscript, *"Attention and Comorbidity-Suppression Modules for Geriatric Pneumonia
Detection on Chest Radiographs: A Multi-Seed Ablation Study of DenseNet-121,"* is currently under review
at *PeerJ*. See [`CITATION.cff`](CITATION.cff).

## Headline finding

All four architectural variants (base, +PAAG, +CSM, +PAAG+CSM) were trained from **5 independent random
seeds** (42, 101, 2024, 7, 2025). Point estimates from a single seed can look like a clear win for PAAG
and/or CSM; pooling across seeds with a DerSimonian–Laird random-effects model (HKSJ-adjusted) does not
establish a robust discriminative benefit for either module. See `analysis/cross_seed_analysis.py` and
`analysis/cross_seed_synthesis.csv` for the seed-by-seed numbers and the pooled estimate.

## Repository layout

```
data/                    Patient-level split assignments, cohort description, label mapping
results/                 Metrics, figures, and statistical-comparison tables for every variant/seed
                         (model checkpoints are NOT included here — see "Trained weights" below)
analysis/                Standalone scripts for the cross-seed statistical synthesis
notebooks/               The Jupyter notebooks used for training, evaluation, and this repo's analysis
data_preparation/        Scripts/notes for building the geriatric cohort from the source datasets
requirements.txt         Pinned package versions from the environment this was run in
```

## Model variants

| Key | PAAG | CSM | Description |
|---|---|---|---|
| `base` | off | off | DenseNet-121, no additions |
| `paag` | on | off | + Pathology-Aware Attention Gate |
| `csm` | off | on | + Comorbidity Suppressor Module |
| `paag_csm` | on | on | Full model |

## Data

This repository does **not** redistribute patient images. It uses two public, de-identified datasets:

- **CheXpert** (Stanford ML Group) — https://stanfordmlgroup.github.io/competitions/chexpert/
- **ChestX-ray8 / ChestX-ray14** (NIH Clinical Center) — https://nihcc.app.box.com/v/ChestXray-NIHCC

`data/split_assignments.csv` records, for every image used in this study, which split (train/validation/test)
it was assigned to, its diagnostic group, source dataset, and basic demographics — so results here can be
reproduced exactly against your own copy of the two source datasets. `data/label_mapping.csv` documents how
each dataset's folder-level label maps to this study's six diagnostic groups and the binary pneumonia target.

## Reproducing

1. Obtain CheXpert and ChestX-ray8/14 directly from their providers (registration required) and arrange
   images into the folder structure expected by `notebooks/` (see `data_preparation/`).
2. `pip install -r requirements.txt`
3. Run the training notebook. Each (architecture variant × seed) is one independent training run; this
   project was developed on Paperspace/Gradient free-tier GPUs, which cap sessions at 6 hours, so the
   notebook trains and caches one (variant, seed) combination at a time and is safe to stop and resume
   across sessions.
4. Run the analysis notebook (or `analysis/cross_seed_analysis.py` directly against a `results/` folder)
   to reproduce the statistical tables.

## Trained weights

The 20 trained checkpoints (4 variants × 5 seeds) are hosted separately on Zenodo rather than in this
repository: **[INSERT ZENODO DOI]**. See that archive's own README for loading instructions, input
preprocessing, and calibration caveats.

## Citing this work

See [`CITATION.cff`](CITATION.cff). Please cite the manuscript once it has a DOI; until then, cite the
software repository itself via its own `CITATION.cff` entry.

## License

- **Code** in this repository: MIT License (see [`LICENSE`](LICENSE)).
- **Trained weights** (Zenodo archive): CC BY 4.0.
- **Data**: not redistributed here; CheXpert and ChestX-ray8/14 remain subject to their original
  providers' data use agreements.

## Contact

Isaac Wafula Wekesa, School of Computing and Information Technology, Jomo Kenyatta University of
Agriculture and Technology (JKUAT), Nairobi, Kenya.
