# ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction

This repository accompanies the paper:

**ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction**

It provides a **fully reproducible pipeline** for molecular property prediction using ArcMol, covering:

- Molecular representation preparation
- Task-adaptive ArcMol training
- Optuna-based hyperparameter search
- Test-only inference from trained models
- Extraction of **learned task-adaptive spherical representations**
- Batch evaluation and metric reporting

---

## Visual Overview

<img src="TOC/TOC.png" width="1028">

*Conceptual overview of ArcMol. The framework learns task-adaptive spherical latent representations for molecular property prediction.*

<br><br>

<img src="TOC/ARCMOL.png" width="1028">

*Detailed illustration of ArcMol, showing task-adaptive feature fusion, spherical projection, and downstream prediction.*

---


## Overview of Environments

This project uses **two separate conda environments**, corresponding to two distinct stages.

### 1. `cmd_fp` — Molecular Representation Preparation

This environment is used to:
- Prepare initial molecular representations (e.g. fingerprints, pretrained embeddings)
- Generate `.pkl` files that serve as **inputs to ArcMol training**

Typical outputs:
- `*_train.pkl`
- `*_valid.pkl`
- `*_test.pkl`

Environment file:
```text
cmd_fp.yml
```

> ⚠️ The code for molecular representation preparation is **not included** in this repository, as it may depend on external toolkits or proprietary pipelines.

---

### 2. `cmd_arcmol` — ArcMol Training & Analysis

This environment is used to:
- Train ArcMol models
- Run Optuna hyperparameter search
- Perform inference and evaluation
- Extract **task-adaptive hidden representations (Z)** from trained models

Environment file:
```text
cmd_arcmol.yml
```

---

## Repository Structure

```text
arcmol/
├── src/
│   ├── attention_pooling_fusion.py     # ArcMol attention pooling module
│   ├── main_arcmol_mcc_r2.py           # ArcMol training (MCC / R2)
│   ├── optuna_arcmol_search.py         # Optuna (single task)
│   ├── optuna_batch_tasks.py           # Optuna (batch tasks)
│   ├── test_only_arcmol.py             # Test-only inference
│   └── extract_features_z.py           # Extract learned ArcMol Z representations
│
├── scripts/
│   ├── batch_test.py                   # Batch testing
│   └── generate_report.py              # Metric summary generation
│
├── data/
│   ├── bbb_logbb_train.pkl             # Classification example (BBB)
│   ├── bbb_logbb_valid.pkl
│   ├── bbb_logbb_test.pkl
│   ├── CHEMBL2147_Ki_train.pkl          # Regression example (CHEMBL2147)
│   └── CHEMBL2147_Ki_test.pkl
│
├── checkpoints/
│   ├── bbb_logbb/
│   │   ├── model.bundle.pt
│   │   └── model.pth
│   └── CHEMBL2147_Ki/
│       ├── model.bundle.pt
│       └── model.pth
│
├── configs/
│   ├── tasks_template_admet.csv
│   └── tasks_template_moleculeACE.csv
│
├── cmd_fp.yml
├── cmd_arcmol.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Environment Setup

### Create `cmd_fp`

```bash
conda env create -f cmd_fp.yml
conda activate cmd_fp
```

### Create `cmd_arcmol`

```bash
conda env create -f cmd_arcmol.yml
conda activate cmd_arcmol
```

---

## Step 1: Prepare Input Molecular Features

Using the **cmd_fp** environment, prepare molecular representations and save them as `.pkl` files.

These `.pkl` files are **inputs to ArcMol**, and are *not* produced by ArcMol itself.

Each `.pkl` file should correspond to one data split:

```text
{task_name}_train.pkl
{task_name}_valid.pkl   # optional
{task_name}_test.pkl
```

The exact feature extraction method (e.g., fingerprints, pretrained encoders) is **outside the scope of this repository**.

---

## Step 1: Prepare Molecular Representation `.pkl` Files (cmd_fp)

ArcMol **consumes precomputed** molecular representation files in `.pkl` format:

- `{task_name}_train.pkl`
- `{task_name}_valid.pkl` (optional)
- `{task_name}_test.pkl`

These `.pkl` files are generated in the **`cmd_fp`** environment using a *separate feature-generation pipeline* (e.g., fingerprints / pretrained embeddings).
> Note: The feature-generation code is **not included** in this repository; please use your existing pipeline to produce the required `{task_name}_{split}.pkl` files.

After you have produced the `.pkl` files, place them under `data/` (or any directory you will pass as `--data_dir`).

---

## Step 2:
## Step 2: ArcMol Training (with Hyperparameter Optimization)

ArcMol models are trained in the **`cmd_arcmol`** environment.
Training can be performed in **two modes**:

1. **Fixed-parameter training** (manual hyperparameters)
2. **Optuna-based hyperparameter optimization** (recommended, used in the paper)

---

### 2.1 Fixed-parameter Training

This mode runs a single training job with user-specified hyperparameters.

#### Classification Example (BBB)

```bash
conda activate cmd_arcmol
python src/main_arcmol_mcc_r2.py \
  --data_dir data \
  --task_name bbb_logbb \
  --task_type cls \
  --target_name label \
  --epochs 1000 \
  --batch_size 64
```

#### Regression Example (CHEMBL2147 Ki)

```bash
python src/main_arcmol_mcc_r2.py \
  --data_dir data \
  --task_name CHEMBL2147_Ki \
  --task_type reg \
  --target_name Ki \
  --epochs 1000 \
  --batch_size 64
```

Training outputs:
- `*.pth`: model checkpoint
- `*.bundle.pt`: self-contained inference bundle (used for testing and feature extraction)

---

### 2.2 Optuna-based Hyperparameter Optimization 

**ArcMol models are trained using Optuna** to select task-adaptive hyperparameters.

#### Single-task Optimization

```bash
conda activate cmd_arcmol
python src/optuna_arcmol_search.py \
  --data_dir data \
  --task_name bbb_logbb \
  --task_type cls \
  --target_name label \
  --n_trials 50 \
  --save_root arcmol_study_runs/bbb_logbb
```

For each Optuna trial:
- A full ArcMol training run is executed
- Hyperparameters are sampled automatically (learning rate, ArcFace margin/scale, MoE, fusion, regularization)
- Each trial is stored under `trial_XXXX/`

After optimization, the **best-performing trial** is automatically copied to:

```text
arcmol_study_runs/bbb_logbb/best_<metric>/
```

including:
- best model checkpoint (`*.pth`)
- exported inference bundle (`*.bundle.pt`)
- calibration files (if available)

#### Batch-task Optimization

```bash
python src/optuna_batch_tasks.py \
  --tasks_csv configs/tasks_template_admet.csv \
  --n_trials 100 \
  --save_root arcmol_batch_runs
```

In batch mode:
- Each task has an independent Optuna study
- Task type (cls / reg) can be specified or automatically inferred
- Best models are saved to `best_auc/` or `best_rmse/`
- A global summary is written to `batch_best_summary.csv`

---


## Step 3: Test-only Inference

```bash
python src/test_only_arcmol.py \
  --data_dir data \
  --task_name bbb_logbb \
  --bundle checkpoints/bbb_logbb/model.bundle.pt \
  --ckpt checkpoints/bbb_logbb/model.pth \
  --save_preds preds_bbb.csv
```



---
## Step 4: Extract Learned ArcMol Hidden Representations (Z)

`extract_features_z.py` extracts **trained ArcMol hidden-layer representations (Z)** for downstream analysis, using:

- the exported training bundle: `*.bundle.pt`
- the model checkpoint: `*.pth`
- the original dataset splits: `{task_name}_{split}.pkl`

### CLI Arguments

- `--data_dir`: directory containing `{task_name}_{split}.pkl`
- `--task_name`: task name prefix (e.g., `bbb_logbb`)
- `--bundle`: `*.bundle.pt` exported during training
- `--ckpt`: optional; override `ckpt_path` inside bundle
- `--output_dir`: directory to save extracted Z features (`.pkl`)
- `--batch_size`: inference batch size
- `--splits`: which splits to process (`train`, `valid`, `test`)

### Example

```bash
conda activate cmd_arcmol
python src/extract_features_z.py \
  --data_dir data \
  --task_name bbb_logbb \
  --bundle checkpoints/bbb_logbb/model.bundle.pt \
  --ckpt checkpoints/bbb_logbb/model.pth \
  --output_dir z_features_output \
  --batch_size 128 \
  --splits train valid test
```

The script will write split-wise Z features as `.pkl` files into `--output_dir`.

---

## Step 5: Batch Testing and Reporting

```bash
python scripts/batch_test.py
python scripts/generate_report.py
```

Outputs:
- `batch_test_results/*.csv`
- `my_model_summary.csv`

---

## Reproducibility Notes

- Initial molecular features and ArcMol representations are **clearly separated**
- Z features are always extracted **after training**
- Bundles ensure consistent inference across runs and environments

---

## Citation

If you use this code, please cite:

```bibtex
@article{arcmol,
  title={ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction},
  author={},
  journal={},
  year={}
}
```

---

## License

This repository is released for **research use only**.
