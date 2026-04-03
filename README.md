# ArcMol: Task-Adaptive Spherical Representation Learning for Molecular Property Prediction

Official code companion to the paper **ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction**.

**Repository:** [github.com/zyrlia1018/ArcMol](https://github.com/zyrlia1018/ArcMol)

This repository provides a reproducible workflow for molecular property prediction with ArcMol, including:

- Molecular featurization via **`DeepFP_Prep/`** (CSV → multi-representation `.pkl`; see [Step 1b](#step-1b-deepfp_prep--multi-representation-pkls-from-csv))
- Task-adaptive ArcMol training and Optuna-based hyperparameter search
- Test-time inference and batch evaluation
- **Label-free, multi-endpoint ADMET-style prediction** from merged or chunked DeepFP PKLs ([Step 3b](#step-3b-label-free-prediction-from-deepfp-pkls), [`admet_predict_from_fp_pkl.ipynb`](notebook/admet_predict_from_fp_pkl.ipynb))
- **MoleculeNet benchmark replay** (BACE, Clintox, ESOL) via [`benchmark_molecule_net_bace_clintox_esol.ipynb`](notebook/benchmark_molecule_net_bace_clintox_esol.ipynb)
- Extraction of **task-adaptive spherical latent representations** for downstream analysis

---

## Architecture

<p align="center">
  <img src="toc/ARCMOL.jpg" width="400" alt="ArcMol framework overview">
</p>

*ArcMol performs task-adaptive fusion of molecular inputs, spherical projection in latent space, and property prediction. The figure summarizes feature fusion, the spherical bottleneck, and the prediction head.*

---

## Computing Environments

The pipeline uses **two conda environments**: one for featurization and one for ArcMol training, inference, and notebooks.

### 1. `cmd_fp` — molecular representation preparation

**Role**

- Build initial molecular representations (fingerprints, language-model and graph embeddings, optional RDKit descriptors).
- Emit split-wise **`.pkl`** files that serve as **inputs to ArcMol** (ArcMol does not compute these features internally).

**Typical artifacts**

- `{task}_train.pkl`, `{task}_valid.pkl`, `{task}_test.pkl`

**Environment file:** [`cmd_fp.yml`](cmd_fp.yml)  
Create and activate: see [Environment setup](#environment-setup).

> **Note (GitHub vs Zenodo).** Training and inference code for ArcMol lives in this repository. Because featurization involves **large checkpoints** and **heavy, model-specific dependencies**, the **full experimental package** (preprocessed datasets, precomputed feature pickles, aligned asset bundles, and **ADMET ArcMol task weights**) is distributed via Zenodo: [doi:10.5281/zenodo.18972759](https://doi.org/10.5281/zenodo.18972759).

**Zenodo archive (summary)**

| Archive | Contents |
|--------|----------|
| `datasets_processed.tar.gz` | Raw inputs and preprocessed `.pkl` files with molecular fingerprints and fused features used in the study. |
| `FP_set.tar.gz` | Featurization source tree and **pre-trained** backbone weights aligned with the paper setup. |

A **Docker image** is under development to simplify deployment of the featurization stack (multiple deep learning back ends). Until it is published, use `cmd_fp.yml` and the [DeepFP_Prep](DeepFP_Prep/README.md) documentation.

---

### 2. `cmd_admet` — ArcMol training, inference, analysis, and notebooks

**Role**

- Train ArcMol models and run Optuna studies.
- Run held-out inference and evaluation.
- Run **label-free multi-checkpoint prediction** and **benchmark notebooks**.
- Export **task-adaptive hidden representations (Z)** from trained checkpoints.

**Environment file:** [`cmd_admet.yml`](cmd_admet.yml)  

> If your clone still uses `cmd_arcmol.yml`, it is the same role; prefer `cmd_admet.yml` for consistency with this repo.

---

## Repository layout

```text
arcmol/
├── DeepFP_Prep/                    # CSV → chunked PKLs (fingerprints, LMs, SSL GNNs, Uni-Mol, optional RDKit descriptors)
│   ├── feature_process.py
│   ├── embed.py
│   ├── read_pkls.py
│   ├── models/ , utils/
│   └── utils/assets/WEIGHTS_DOWNLOAD_LIST.txt   # checkpoint layout (large weights not in Git)
│
├── src/
│   ├── attention_pooling_fusion.py
│   ├── main_arcmol_mcc_r2.py           # training (classification / regression)
│   ├── optuna_arcmol_search.py         # Optuna, single task
│   ├── optuna_batch_tasks.py           # Optuna, task list from CSV
│   ├── test_only_arcmol.py             # inference on split PKLs (with labels)
│   ├── predict_arcmol_from_fp_pkls.py  # label-free inference: DeepFP PKL dir or merged PKL → wide CSV (all checkpoints)
│   └── extract_features_z.py           # export learned Z representations
│
├── scripts/
│   ├── batch_test.py                   # batch evaluation driver
│   ├── generate_report.py              # aggregate metrics
│   ├── eval_best_from_summary.py       # (if shipped) TEST replay from best_summary.json — used by benchmark notebook
│   ├── assemble_bundle_from_ckpt.py
│   ├── assemble_bundle_from_best_summary.py
│   ├── scan_lipophilicity_rmse.py
│   ├── pack_admet_checkpoints.sh       # tar checkpoints/ (excludes benchmark_* by default)
│   ├── pack_checkpoints_for_github.py  # optional: split checkpoints into several .tar.gz parts
│   ├── pack_checkpoints_split_github.sh
│   └── sync_admet_checkpoints_from_best.sh   # optional: rsync full ADMET tree into checkpoints/
│
├── notebook/
│   ├── deepfp_allowed_14_demo.ipynb              # DeepFP_Prep: 14 allowed embeddings × 3 SMILES (Step 1b)
│   ├── admet_predict_from_fp_pkl.ipynb           # ADMET: merged DeepFP PKL → all endpoints; Zenodo weights; reviewer-facing tables
│   ├── benchmark_molecule_net_bace_clintox_esol.ipynb   # MoleculeNet: BACE / Clintox / ESOL TEST replay vs paper numbers
│   ├── latent_rep_z_cls.ipynb                    # Z visualization, classification (Step 4)
│   └── latent_rep_z_reg.ipynb                    # Z visualization, regression (Step 4)
│
├── releases/
│   └── checkpoints_github_parts/       # MANIFEST.txt (+ optional split tarballs); large *.tar.gz usually gitignored
│
├── data/                               # small examples (BBB, CHEMBL2147 Ki); benchmark tasks may ship *_test.pkl only
├── checkpoints/                        # trained bundles per task (populate from Zenodo; not fully in Git)
├── configs/                            # task list templates (e.g. tasks_template_admet.csv)
├── toc/                                # figures for README
├── cmd_fp.yml
├── cmd_admet.yml
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore                          # typically ignores predictions/, large release tarballs
```

---

## Environment setup

### `cmd_fp`

```bash
conda env create -f cmd_fp.yml
conda activate cmd_fp
```

### `cmd_admet` (ArcMol + notebooks)

```bash
conda env create -f cmd_admet.yml
conda activate cmd_admet
```

---

## Step 1: Input molecular features (`.pkl`)

Molecular representations must be prepared **before** ArcMol training, using the **`cmd_fp`** environment and any pipeline compatible with the expected schema.

**Naming convention (per split)**

```text
{task_name}_train.pkl
{task_name}_valid.pkl    # optional
{task_name}_test.pkl
```

The **`DeepFP_Prep/`** workflow below is the supported in-repo option; externally generated PKLs are valid if they match ArcMol’s input contract.

**Example (shipped in `data/`):** `bbb_logbb_train.pkl` / `*_valid.pkl` / `*_test.pkl` and `CHEMBL2147_Ki_{train,test}.pkl` follow the naming convention above and match the training commands in [Step 2](#step-2-arcmol-training).

---

## Step 1b: DeepFP_Prep — multi-representation PKLs from CSV

**DeepFP_Prep/** is a **lightweight** featurization package: code and small assets ship in Git; **large pre-trained weights are excluded** and must be downloaded separately (or supplied via **`DEEPFP_ASSETS_DIR`**).

**Behavior**

- Ingest a CSV with at least a SMILES column (optional `y`, `split`, and extra columns).
- Canonicalize SMILES and compute **multiple embeddings** registered in `DeepFP_Prep/embed.py` (RDKit fingerprints, MolT5, BioT5, SSL graph encoders, Uni-Mol, …), subject to installed back ends and local checkpoints.
- Optionally attach **`rdkit_descriptors`** per compound.
- Write **chunked** `*_batch_*.pkl` dictionaries (`row_id → record`); merge or convert to `{task}_{split}.pkl` for training.

**Pre-trained checkpoints**

Follow the same organization as [MolRetrieval — Prepare model checkpoints](https://github.com/linhaowei1/MolRetrieval#getting-started): paths must agree with `DeepFP_Prep/utils/env_utils.py`, or point **`DEEPFP_ASSETS_DIR`** at a full asset tree (e.g. from Zenodo **`FP_set`**).

- **Language models (Hugging Face):** `huggingface-cli` ([guide](https://huggingface.co/docs/huggingface_hub/guides/download)). Map repos to folders: `DeepChem/ChemBERTa-77M-MTR` → `ChemBERTa-77M-MTR/`, `QizhiPei/biot5-base` → `BioT5/`, `laituan245/molt5-base` → `MolT5/`. Direct weight files: [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/resolve/main/pytorch_model.bin) · [BioT5](https://huggingface.co/QizhiPei/biot5-base/resolve/main/pytorch_model.bin) · [MolT5](https://huggingface.co/laituan245/molt5-base/resolve/main/pytorch_model.bin).
- **SSL graph encoders:** [GraphMVP Google Drive](https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6?usp=sharing) (see [GraphMVP](https://github.com/chao1224/GraphMVP)). Rename: `Motif.pth` → `Motif/model.pth`, `AM.pth` → `AM/model.pth`, `GPT_TNN.pth` → `GPT_GNN/model.pth`, `GraphCL.pth` → `GraphCL/model.pth`. For the GraphMVP checkpoint, unpack `GraphMVP_complate_features_for_regression.zip` and copy `…/GraphMVP/pretraining_model.pth` → `GraphMVP/model.pth`.
- **MolCLR:** [official `model.pth`](https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth) → `MolCLR/model.pth`.
- **Uni-Mol V1:** [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models) — e.g. [mol_pre_no_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_no_h_220816.pt), optional [mol_pre_all_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt), [mol.dict.txt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt) under `UniMolV1/`.
- **Uni-Mol V2:** [modelzoo](https://huggingface.co/dptech/Uni-Mol2/tree/main/modelzoo) — [84M](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt) · [310M](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/310M/checkpoint.pt) · [1.1B](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/1.1B/checkpoint.pt) under `UniMolV2/{84M,310M,1.1B}/`.
- **Full bundle (optional):** [Zenodo](https://doi.org/10.5281/zenodo.18972759) (includes `FP_set.tar.gz`).

**Documentation**

- Compact index: `DeepFP_Prep/utils/assets/WEIGHTS_DOWNLOAD_LIST.txt`
- Tables, links, and `feature_process.py` flags (`EMBEDDING_MODE`, `--list-embeddings`): **[`DeepFP_Prep/README.md`](DeepFP_Prep/README.md)**

With **`cmd_fp`** active, run featurization from `DeepFP_Prep/`.

**Example**

```bash
conda activate cmd_fp
cd DeepFP_Prep
python feature_process.py --list-embeddings   # optional: embeddings available in this env
# set EMBEDDING_MODE in feature_process.py ("all" vs "allowed"), then:
python feature_process.py
```

Chunked `*_batch_*.pkl` files can be merged or converted to `{task}_{split}.pkl` for ArcMol (see [`DeepFP_Prep/README.md`](DeepFP_Prep/README.md)).

**Worked example — notebook (14 `allowed` embeddings)**

End-to-end demo with **`cmd_fp` active**: after checkpoints are present under `DeepFP_Prep/utils/assets` (or `DEEPFP_ASSETS_DIR`), open and run:

| Notebook | What it does |
|----------|----------------|
| [`notebook/deepfp_allowed_14_demo.ipynb`](notebook/deepfp_allowed_14_demo.ipynb) | Three random SMILES → all **14** names in `ALLOWED_EMBEDDINGS` ∩ `Embedding.available()` + RDKit descriptors → chunked PKL under `notebook/_demo_pkls_output/`; summary table of vector dimensions. |

Edit `REPO_ROOT` in the first cells if your clone path is not auto-detected. Outputs in the notebook are in English.

**中文摘要：** `DeepFP_Prep` 从 CSV 生成多路分子表征；预计算特征与完整实验资源见 Zenodo。大权重按上文与 `env_utils.py` 部署，或使用 `FP_set` / `DEEPFP_ASSETS_DIR`。细节见 `DeepFP_Prep/README.md` 与 `WEIGHTS_DOWNLOAD_LIST.txt`。分步演示见上表 `deepfp_allowed_14_demo.ipynb`。

---

## Step 2: ArcMol training

Use the **`cmd_admet`** environment. Two modes are supported:

1. **Fixed hyperparameters** — single run with user-specified settings.
2. **Optuna search** — recommended configuration for the paper; explores learning rate, ArcFace margin/scale, MoE, fusion, and regularization.

### 2.1 Fixed hyperparameters

#### Classification (BBB)

```bash
conda activate cmd_admet
python src/main_arcmol_mcc_r2.py \
  --data_dir data \
  --task_name bbb_logbb \
  --task_type cls \
  --target_name label \
  --epochs 1000 \
  --batch_size 64
```

#### Regression (CHEMBL2147 Ki)

```bash
python src/main_arcmol_mcc_r2.py \
  --data_dir data \
  --task_name CHEMBL2147_Ki \
  --task_type reg \
  --target_name Ki \
  --epochs 1000 \
  --batch_size 64
```

**Outputs**

- `*.pth` — model checkpoint  
- `*.bundle.pt` — self-contained bundle for inference and feature extraction  

**Example:** The BBB and CHEMBL2147 commands above use the small **`data/`** tree shipped in this repository (`--data_dir data`, tasks `bbb_logbb` / `CHEMBL2147_Ki`).

### 2.2 Optuna-based optimization

#### Single task

```bash
conda activate cmd_admet
python src/optuna_arcmol_search.py \
  --data_dir data \
  --task_name bbb_logbb \
  --task_type cls \
  --target_name label \
  --n_trials 50 \
  --save_root arcmol_study_runs/bbb_logbb
```

Each trial runs a full training job; artifacts are stored under `trial_XXXX/`. The best trial is copied to:

```text
arcmol_study_runs/bbb_logbb/best_<metric>/
```

including the top checkpoint (`.pth`), exported bundle (`.bundle.pt`), and calibration files when applicable.

#### Batch tasks (CSV-driven)

```bash
python src/optuna_batch_tasks.py \
  --tasks_csv configs/tasks_template_admet.csv \
  --n_trials 100 \
  --save_root arcmol_batch_runs
```

Each row defines an independent study; task type (`cls` / `reg`) may be given or inferred. Best models are written under `best_auc/` or `best_rmse/`, with a global summary in `batch_best_summary.csv`.

**Example:** Point `--data_dir` / `--tasks_csv` at the same layout as your Step 2 runs; `configs/tasks_template_admet.csv` is a starting template for batch mode.

---

## Step 3: Test-only inference (split PKLs with labels)

```bash
conda activate cmd_admet
python src/test_only_arcmol.py \
  --data_dir data \
  --task_name bbb_logbb \
  --bundle checkpoints/bbb_logbb/model.bundle.pt \
  --ckpt checkpoints/bbb_logbb/model.pth \
  --save_preds preds_bbb.csv
```

**Example:** Uses bundled **`data/`** splits and **`checkpoints/bbb_logbb/model.bundle.pt`** / **`model.pth`**; adjust paths for your own tasks.

---

## Step 3b: Label-free prediction from DeepFP PKLs

[`src/predict_arcmol_from_fp_pkls.py`](src/predict_arcmol_from_fp_pkls.py) runs **label-free** inference when you already have DeepFP-style PKLs (chunked `*_batch_*.pkl` **or** a **single merged** pickle: top-level `dict` of `row_id → record` with `SMILES`, `rdkit_descriptors`, and the same embedding keys as training).

**Checkpoints**

- Download **ADMET / multi-endpoint** trained bundles from **Zenodo** ([doi:10.5281/zenodo.18972759](https://doi.org/10.5281/zenodo.18972759)) and extract into **`checkpoints/`** so each task is `checkpoints/<task_name>/` with a matching `*.bundle.pt` + same-stem `*.pth`.
- Expected task names are listed in [`configs/tasks_template_admet.csv`](configs/tasks_template_admet.csv) and in [`releases/checkpoints_github_parts/MANIFEST.txt`](releases/checkpoints_github_parts/MANIFEST.txt).

**CLI examples**

```bash
conda activate cmd_admet
cd /path/to/ArcMol

# All tasks under checkpoints/: either chunked PKL directory OR one merged file
python src/predict_arcmol_from_fp_pkls.py \
  --pkl_dir /path/to/your_deepfp_pkls \
  --checkpoints_root checkpoints \
  --out_dir predictions \
  --out_csv admet_all_endpoints_preds.csv

python src/predict_arcmol_from_fp_pkls.py \
  --pkl_file data/_demo_pkls_output/all_batch_0_3.pkl \
  --checkpoints_root checkpoints \
  --out_dir predictions \
  --out_csv admet_all_endpoints_preds.csv \
  --no_skip_benchmark
```

**Behavior**

- **`--mode all_checkpoints` (default):** every subdirectory of `checkpoints/` that contains a paired `*.bundle.pt` and `*.pth` is run; results are **one wide CSV** (one row per compound, one or more columns per endpoint). Classification tasks add `_pred_prob` / `_pred_label`; regression adds `_pred`.
- **`--no_skip_benchmark`:** include `benchmark_*` folders (default CLI skips them; use this to match “scan everything”).
- **`--exclude_dir_prefix`:** repeatable; skip folder names with a given prefix.
- **`--mode single`:** pass `--bundle` / `--ckpt` for one task only.

**Default output directory** in the script is **`预测/`** (Chinese folder name). You can set **`--out_dir predictions`** for an English path (recommended for sharing).

**Notebook (reviewer-friendly):** [`notebook/admet_predict_from_fp_pkl.ipynb`](notebook/admet_predict_from_fp_pkl.ipynb) sets `sys.path` to `src/`, loads a merged demo PKL, runs all checkpoints, writes CSV under `predictions/`, and **embeds summary tables in the notebook** for supplementary / peer review. **Restart kernel & run all** after downloading Zenodo weights.

PKL keys must match each bundle’s `fusion_embed_types` and RDKit attribute set. See the script docstring for full options.

**中文说明：** 无标签时，可用 DeepFP 分块目录或单个合并 pkl，对 `checkpoints/` 下全部成对 bundle+权重一次性宽表推理；权重请从 Zenodo 下载到 `checkpoints/`。

---

## Step 3c: MoleculeNet benchmark notebook (BACE, Clintox, ESOL)

| Notebook | What it does |
|----------|----------------|
| [`notebook/benchmark_molecule_net_bace_clintox_esol.ipynb`](notebook/benchmark_molecule_net_bace_clintox_esol.ipynb) | Replays **held-out TEST** evaluation for **BACE**, **Clintox**, **ESOL** using `checkpoints/benchmark_bace`, `benchmark_clintox`, `benchmark_esol` (`best_summary.json`, `model.pth`, `calib.json` for classification, optional `eval_preprocess.pkl` when only `*_test.pkl` is shipped). Compares to paper-reported metrics in the notebook. Expects `scripts/eval_best_from_summary.py` if your layout uses that entry point — adjust `REPO_ROOT` / paths in the first cells. |

Use the same environment as ArcMol (**GPU recommended** for parity with saved `best_summary.json`).

---

## Step 4: Extract learned representations (Z)

`extract_features_z.py` exports **ArcMol hidden states (Z)** using the training bundle, checkpoint, and split PKLs.

**Main arguments**

| Argument | Description |
|----------|-------------|
| `--data_dir` | Directory with `{task_name}_{train|valid|test}.pkl` |
| `--task_name` | Task prefix (e.g. `bbb_logbb`) |
| `--bundle` | `*.bundle.pt` from training |
| `--ckpt` | Optional override for weights inside the bundle |
| `--output_dir` | Where to save Z features (`.pkl`) |
| `--batch_size` | Inference batch size |
| `--splits` | Subset to process: `train`, `valid`, `test` |

**Example**

```bash
conda activate cmd_admet
python src/extract_features_z.py \
  --data_dir data \
  --task_name bbb_logbb \
  --bundle checkpoints/bbb_logbb/model.bundle.pt \
  --ckpt checkpoints/bbb_logbb/model.pth \
  --output_dir z_features_output \
  --batch_size 128 \
  --splits train valid test
```

Split-wise Z matrices are written under `--output_dir`.

**Worked example — interpreting and visualizing Z**

The notebooks below run end-to-end: they load (or reproduce) **unit-normalized latent Z**, then show **t-SNE**, a **2D spherical map**, and a **3D sphere projection**—useful for qualitative inspection of the task-adaptive spherical representation.

| Notebook | Task type | Illustrative task |
|----------|-----------|-------------------|
| [`notebook/latent_rep_z_cls.ipynb`](notebook/latent_rep_z_cls.ipynb) | Classification | e.g. BBB (`bbb_logbb`) |
| [`notebook/latent_rep_z_reg.ipynb`](notebook/latent_rep_z_reg.ipynb) | Regression | e.g. `CHEMBL2147_Ki` |

Open them from the repo root (or set `sys.path` as in the first cells) and edit the bundled paths (`DATA_DIR`, `BUNDLE_PATH`, `CKPT_PATH`, `OUT_DIR`) to match your artifacts.

---

## Step 5: Batch testing and reporting

```bash
conda activate cmd_admet
python scripts/batch_test.py
python scripts/generate_report.py
```

**Outputs:** `batch_test_results/*.csv`, `my_model_summary.csv`

**Example:** In `scripts/batch_test.py`, set `TASKS_CSV_PATH` (must include `task_name` and `data_dir` columns), `CHECKPOINTS_ROOT`, and `OUTPUT_ROOT`, then run the two commands from the repository root with **`cmd_admet`** active. Use `scripts/generate_report.py` to aggregate metrics into `my_model_summary.csv` (edit paths inside the script if needed).

---

## Helper scripts (checkpoints & releases)

| Script | Purpose |
|--------|---------|
| [`scripts/pack_admet_checkpoints.sh`](scripts/pack_admet_checkpoints.sh) | Create a single `tar.gz` of `checkpoints/` (excludes `benchmark_*` by default) for backup or mirroring. |
| [`scripts/pack_checkpoints_for_github.py`](scripts/pack_checkpoints_for_github.py) | Optional: split `checkpoints/` into multiple `.tar.gz` parts (see `--help`). Regenerates `MANIFEST.txt` with Zenodo header. |
| [`scripts/pack_checkpoints_split_github.sh`](scripts/pack_checkpoints_split_github.sh) | Shell wrapper for the Python packer. |
| [`scripts/sync_admet_checkpoints_from_best.sh`](scripts/sync_admet_checkpoints_from_best.sh) | Optional: `rsync` a local “full ADMET” tree into `checkpoints/`. |

---

## Reproducibility

- **Featurization vs ArcMol:** raw or engineered molecular inputs are produced upstream; ArcMol consumes fixed `.pkl` schemas.
- **Z extraction:** always run **after** training, from the exported bundle and checkpoint aligned with that run.
- **Bundles:** `.bundle.pt` pins preprocessing and fusion metadata so inference and Z export stay consistent across machines.
- **ADMET wide prediction:** Zenodo weights + merged/chunked DeepFP PKLs aligned with training embeddings; rerun notebooks with **Kernel → Restart & Run All** before sharing executed `.ipynb` with reviewers.

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

This project is licensed under the [MIT License](LICENSE).
