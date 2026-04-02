# ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction

This repository accompanies the paper:

**ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction**

It provides a **fully reproducible pipeline** for molecular property prediction using ArcMol, covering:

- Molecular representation preparation (**`DeepFP_Prep/`**: CSV → multi-representation PKLs, see below)
- Task-adaptive ArcMol training
- Optuna-based hyperparameter search
- Test-only inference from trained models
- Extraction of **learned task-adaptive spherical representations**
- Batch evaluation and metric reporting

---

## Architecture

<p align="center">
  <img src="toc/ARCMOL.jpg" width="400">
</p>

*The framework learns task-adaptive spherical latent representations for molecular property prediction.*
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

> ⚠️ The core training and inference logic of ArcMol is hosted in this GitHub repository. However, due to the large file sizes and complex dependencies involved in molecular representation preparation, the complete experimental package is archived on Zenodo(doi: 10.5281/zenodo.18972759)


---
Specifically, the Zenodo archive contains:

**datasets_processed.tar.gz**: Includes all raw data files and the pre-processed .pkl files (with generated molecular fingerprints/features) used in the study.

**FP_set.tar.gz**: Contains the full source code for data processing and representation learning, along with the pre-trained model weights.

Given that the environment involves multiple deep-learning pre-trained models and can be complex to deploy manually, we are actively preparing a Docker image to provide a "plug-and-play" environment for the community. This will ensure seamless reproducibility and ease of use.


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
├── DeepFP_Prep/                        # DeepFP: CSV → chunked PKLs (fingerprints + pretrained embeddings + RDKit descriptors)
│   ├── feature_process.py
│   ├── embed.py
│   ├── read_pkls.py
│   ├── models/ , utils/
│   └── utils/assets/WEIGHTS_DOWNLOAD_LIST.txt   # where to place large checkpoints (not in Git)
│
├── src/
│   ├── attention_pooling_fusion.py     # ArcMol attention pooling module
│   ├── main_arcmol_mcc_r2.py           # ArcMol training (MCC / R2)
│   ├── optuna_arcmol_search.py         # Optuna (single task)
│   ├── optuna_batch_tasks.py           # Optuna (batch tasks)
│   ├── test_only_arcmol.py             # Test-only inference (split PKLs with labels)
│   ├── predict_arcmol_from_fp_pkls.py  # Label-free inference from DeepFP PKL directory → one wide CSV
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

For training, ArcMol still expects split-wise files named `{task_name}_{train|valid|test}.pkl`. You can produce them with **`DeepFP_Prep/`** (see next section) or any compatible pipeline.

---

## Step 1b: DeepFP_Prep — multi-representation PKLs from CSV (included)

This repository ships **`DeepFP_Prep/`**, a **lightweight** feature pipeline (no large pretrained weights in Git) that:

- Reads a CSV with at least a SMILES column (optional `y`, `split`, extra columns).
- Standardizes SMILES and computes **multiple embeddings** (RDKit fingerprints, MolT5, BioT5, SSL graph models, UniMol, … — whatever is registered in `DeepFP_Prep/embed.py` and available in your environment).
- Optionally writes **`rdkit_descriptors`** (full RDKit descriptor dict per molecule).
- Writes **chunked** `*_batch_*.pkl` files (one dict: `row_id → record`), which you can merge or convert to `{task}_{split}.pkl` for training.

**Prepare model checkpoints** (same idea as [MolRetrieval — Getting Started](https://github.com/linhaowei1/MolRetrieval#getting-started)): place files so paths match `DeepFP_Prep/utils/env_utils.py`, or set **`DEEPFP_ASSETS_DIR`** to an existing asset tree (e.g. Zenodo `FP_set`).

- **Language models (Hugging Face):** use `huggingface-cli` ([download guide](https://huggingface.co/docs/huggingface_hub/guides/download)). Example repos: `DeepChem/ChemBERTa-77M-MTR` → folder `ChemBERTa-77M-MTR/`; `QizhiPei/biot5-base` → `BioT5/`; `laituan245/molt5-base` → `MolT5/`. Direct weights: [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/resolve/main/pytorch_model.bin) · [BioT5](https://huggingface.co/QizhiPei/biot5-base/resolve/main/pytorch_model.bin) · [MolT5](https://huggingface.co/laituan245/molt5-base/resolve/main/pytorch_model.bin).
- **SSL graph models:** [GraphMVP Google Drive folder](https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6?usp=sharing) ([GraphMVP repo](https://github.com/chao1224/GraphMVP)). Rename: `Motif.pth` → `Motif/model.pth`, `AM.pth` → `AM/model.pth`, `GPT_TNN.pth` → `GPT_GNN/model.pth`, `GraphCL.pth` → `GraphCL/model.pth`. **GraphMVP:** from `GraphMVP_complate_features_for_regression.zip`, copy `…/GraphMVP/pretraining_model.pth` → `GraphMVP/model.pth`.
- **MolCLR:** [checkpoint `model.pth`](https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth) → `MolCLR/model.pth`.
- **Uni-Mol V1:** [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models) — e.g. [mol_pre_no_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_no_h_220816.pt), optional [mol_pre_all_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt), [mol.dict.txt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt) under `UniMolV1/`.
- **Uni-Mol V2:** [modelzoo tree](https://huggingface.co/dptech/Uni-Mol2/tree/main/modelzoo) — e.g. [84M](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt) · [310M](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/310M/checkpoint.pt) · [1.1B](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/1.1B/checkpoint.pt) under `UniMolV2/{84M,310M,1.1B}/`.
- **Optional full bundle:** [Zenodo record](https://doi.org/10.5281/zenodo.18972759) (includes `FP_set.tar.gz`).

Short index: `DeepFP_Prep/utils/assets/WEIGHTS_DOWNLOAD_LIST.txt`. Full tables and commands: **`DeepFP_Prep/README.md`**.

**Quick start (cmd_fp or a PyTorch+RDKit env):**

```bash
cd DeepFP_Prep
pip install -r requirements-deepfp-prep.txt
python feature_process.py --list-embeddings    # names available on this machine
# Edit EMBEDDING_MODE in feature_process.py: "all" vs "allowed" subset, then:
python feature_process.py
```

More detail: `DeepFP_Prep/README.md`.

**中文摘要：** `DeepFP_Prep` 从 CSV 批量生成多种分子表征（指纹、ChemBERTa/MolT5/BioT5、SSL 图模型、Uni-Mol V1/V2、可选 RDKit 描述符）的 pkl；大权重按上文链接与 `env_utils.py` 放置，或使用 Zenodo `FP_set` / `DEEPFP_ASSETS_DIR`。细则见 `DeepFP_Prep/README.md` 与 `WEIGHTS_DOWNLOAD_LIST.txt`。

---

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

## Step 3b: Label-free prediction from DeepFP PKLs (wide CSV)

If you only have **DeepFP-style chunked PKLs** (a directory of `*_batch_*.pkl` with `SMILES`, `rdkit_descriptors`, and all embeddings used during training — **no labels required**), use:

```bash
conda activate cmd_arcmol
cd /path/to/ArcMol-main
python src/predict_arcmol_from_fp_pkls.py \
  --pkl_dir /path/to/your_deepfp_pkls \
  --checkpoints_root checkpoints
```

- **Default behavior:** scans every `checkpoints/<task>/*.bundle.pt` + matching `*.pth` and writes **one** wide table:  
  **`预测/admet_all_endpoints_preds.csv`** (under the repo root; override with `--out_dir` / `--out_csv`).
- **Single-task debug:** `--mode single --bundle ... --ckpt ...`

Requirements: PKL keys must match each bundle’s `fusion_embed_types` and RDKit attribute set (same as training). See `src/predict_arcmol_from_fp_pkls.py` header for options.

**中文说明：** 无标签时，用 DeepFP 生成的分块 pkl 目录，可对 `checkpoints` 下全部已训练任务一次性推理，结果合并为**单个 CSV**（默认保存在仓库根目录的 `预测/` 文件夹）。

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
