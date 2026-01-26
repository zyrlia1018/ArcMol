# ArcMol: Optuna-based Molecular Property Modeling Toolkit

This repository provides a **reproducible training–evaluation pipeline** for molecular property prediction using **ArcMol** models, with support for:

- Single-task and batch-task **Optuna hyperparameter search**
- Unified **training + inference bundle export**
- **Test-only inference** from trained bundles
- **Batch evaluation** across many tasks
- Automatic **metric report generation** (classification & regression)

The repo is designed for research usage and easy extension.

---

## Features

- 🔬 Molecular representation learning with ArcMol
- 🔁 Optuna-based hyperparameter optimization
- 📦 Self-contained inference bundles (`.bundle.pt`)
- 📊 Batch testing + automatic metric summary
- 🧪 Supports **classification** and **regression** tasks

---

## Repository Structure

```text
arcmol/
├── src/
│   ├── attention_pooling_fusion.py     # Core model component
│   ├── main_arcmol.py                  # Training entry (AUC / RMSE)
│   ├── main_arcmol_mcc_r2.py           # Training entry (MCC / R2)
│   ├── optuna_arcmol_search.py         # Optuna (single task)
│   ├── optuna_batch_tasks.py           # Optuna (batch tasks)
│   └── test_only_arcmol.py             # Inference from bundle
│
├── scripts/
│   ├── batch_test.py                   # Batch inference runner
│   └── generate_report.py              # Metric summary generator
│
├── configs/
│   ├── tasks_template_admet.csv
│   ├── tasks_template_moleculeACE.csv
│   └── tasks_example.csv               # Example task list (no real paths)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Create a clean Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Minimum required packages:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `optuna`

---

## Data Format

Each task directory should contain:

```text
data_dir/
├── <task_name>_train.pkl
├── <task_name>_valid.pkl   # optional
└── <task_name>_test.pkl
```

If `*_valid.pkl` is missing, validation data will be split from training.

---

## Usage

### 1. Single-task Optuna Search

```bash
python src/optuna_arcmol_search.py \
  --data_dir /path/to/data_dir \
  --task_name TASK_NAME \
  --task_type cls \
  --target_name target_y \
  --n_trials 50 \
  --save_root runs/optuna_single
```

Best checkpoints and bundles are copied to:

```text
runs/optuna_single/best_<metric>/
```

---

### 2. Batch-task Optuna Search

Prepare a CSV file with columns:

- `task_name` (required)
- `data_dir` (required)
- `target_name` (required)
- `task_type` (optional: cls / reg)
- `dataset` (optional)

Example:

```csv
task_name,data_dir,target_name,task_type
CHEMBL123,/path/to/data,y,cls
```

Run:

```bash
python src/optuna_batch_tasks.py \
  --tasks_csv configs/tasks_example.csv \
  --save_root runs/optuna_batch \
  --n_trials 100
```

Summary output:

```text
runs/optuna_batch/batch_best_summary.csv
```

---

### 3. Test-only Inference

```bash
python src/test_only_arcmol.py \
  --data_dir /path/to/data_dir \
  --task_name TASK_NAME \
  --bundle /path/to/model.bundle.pt \
  --ckpt /path/to/model.pth \
  --save_preds preds.csv \
  --extra_attrs SMILES
```

> Note: `--ckpt` is required unless `ckpt_path` is stored inside the bundle.

---

### 4. Batch Testing

Organize checkpoints as:

```text
checkpoints/
└── TASK_NAME/
    ├── model.bundle.pt
    └── model.pth
```

Then run:

```bash
python scripts/batch_test.py
```

Predictions are written to:

```text
batch_test_results/
```

---

### 5. Generate Metric Report

```bash
python scripts/generate_report.py
```

Outputs:

```text
my_model_summary.csv
```

Metrics:
- Classification: AUC, MCC, F1, ACC
- Regression: RMSE, MAE, R2

---

## Git Ignore Policy

The following are **not committed** by default:

- Model artifacts: `*.pt`, `*.pth`, `*.bundle.pt`
- Datasets: `*.pkl`
- Training logs and outputs

This keeps the repository lightweight and reproducible.

---

## Notes

- All scripts are designed to be run from the **project root**
- Paths in example CSVs are placeholders only
- You are encouraged to fork and adapt the pipeline for your own datasets

---

## License

This project is intended for research use.  
Add a license file if you plan to redistribute.

---

If you find this repo useful, feel free to ⭐ star it.
