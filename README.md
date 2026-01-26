# ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction

This repository accompanies the paper:

**ArcMol Enables Task-Adaptive Spherical Representation Learning for Molecular Property Prediction**

It provides a **fully reproducible pipeline** for molecular property prediction using ArcMol, covering:

- Molecular representation extraction
- Task-adaptive ArcMol training
- Optuna-based hyperparameter search
- Test-only inference from trained models
- Batch evaluation and metric reporting

The code is organized to clearly separate **representation extraction** and **ArcMol modeling**, following the experimental setup in the paper.

---

## Overview of Environments

This project uses **two separate conda environments**, corresponding to two distinct stages.

### 1. `cmd_fp` — Molecular Representation Extraction

This environment is used to:
- Generate molecular representations (fingerprints / embeddings)
- Produce intermediate `.pkl` feature files consumed by ArcMol

Typical outputs:
- `*_train.pkl`
- `*_valid.pkl`
- `*_test.pkl`

Environment file:
```text
cmd_fp.yml
```

### 2. `cmd_arcmol` — ArcMol Training & Evaluation

This environment is used to:
- Train ArcMol models
- Run Optuna hyperparameter search
- Perform inference and evaluation

Environment file:
```text
cmd_arcmol.yml
```

> ⚠️ These two environments are intentionally decoupled to ensure reproducibility and modularity.

---

## Repository Structure

```text
arcmol/
├── src/
│   ├── attention_pooling_fusion.py     # ArcMol attention pooling module
│   ├── main_arcmol.py                  # ArcMol training (AUC / RMSE)
│   ├── main_arcmol_mcc_r2.py           # ArcMol training (MCC / R2)
│   ├── optuna_arcmol_search.py         # Optuna (single task)
│   ├── optuna_batch_tasks.py           # Optuna (batch tasks)
│   ├── test_only_arcmol.py             # Test-only inference
│   └── extract_features_z.py           # Extract learned ArcMol representations
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

### Create `cmd_fp` (Feature Extraction)

```bash
conda env create -f cmd_fp.yml
conda activate cmd_fp
```

### Create `cmd_arcmol` (ArcMol)

```bash
conda env create -f cmd_arcmol.yml
conda activate cmd_arcmol
```

---

## Step 1: Molecular Representation Extraction

Using the **cmd_fp** environment, generate molecular representations and save them as `.pkl` files.

Example:

```bash
conda activate cmd_fp
python extract_features_z.py   --input_csv molecules.csv   --output_pkl bbb_logbb_train.pkl
```

The generated `.pkl` files are later consumed directly by ArcMol.

---

## Step 2: ArcMol Training

Switch to the **cmd_arcmol** environment.

### Classification Example (BBB)

```bash
conda activate cmd_arcmol
python src/main_arcmol_mcc_r2.py   --data_dir data   --task_name bbb_logbb   --task_type cls   --target_name label   --epochs 1000   --batch_size 64
```

### Regression Example (CHEMBL2147 Ki)

```bash
python src/main_arcmol.py   --data_dir data   --task_name CHEMBL2147_Ki   --task_type reg   --target_name Ki   --epochs 1000   --batch_size 64
```

Training produces:
- `model.pth` (model checkpoint)
- `model.bundle.pt` (self-contained inference bundle)

---

## Step 3: Test-only Inference

```bash
python src/test_only_arcmol.py   --data_dir data   --task_name bbb_logbb   --bundle checkpoints/bbb_logbb/model.bundle.pt   --ckpt checkpoints/bbb_logbb/model.pth   --save_preds preds_bbb.csv
```

---

## Step 4: Extract Learned ArcMol Representations

After training, ArcMol representations can be extracted for downstream analysis.

```bash
python src/extract_features_z.py   --bundle checkpoints/bbb_logbb/model.bundle.pt   --ckpt checkpoints/bbb_logbb/model.pth   --data_dir data   --task_name bbb_logbb   --output_pkl bbb_logbb_z.pkl
```

These representations correspond to **task-adaptive spherical embeddings** described in the paper.

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

## Notes on Reproducibility

- All datasets and trained models used in the paper are provided
- Feature extraction and modeling are strictly separated
- Bundles ensure consistent inference across environments

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
