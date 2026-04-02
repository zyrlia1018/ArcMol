# DeepFP_Prep

Lightweight **CSV → multi-representation `.pkl`** pipeline for the ArcMol **`cmd_fp`** stage (fingerprints, LMs, SSL GNNs, Uni-Mol, optional RDKit descriptors). Large checkpoints are **not** in Git.

- ArcMol training & label-free batch prediction: root [`README.md`](../README.md) (Step 1b, Step 3b) and `src/predict_arcmol_from_fp_pkls.py`.
- Path layout is defined in [`utils/env_utils.py`](utils/env_utils.py) (or override with **`DEEPFP_ASSETS_DIR`**).

---

## Prepare model checkpoints

*(Aligned with [MolRetrieval — Getting Started → Prepare model checkpoints](https://github.com/linhaowei1/MolRetrieval#getting-started). After downloading, paths must match `utils/env_utils.py`.)*

To run **ChemBERTa**, **BioT5**, **MolT5**, and **SSL-based graph models**, download the corresponding checkpoints.

### Molecule language models (Hugging Face)

Use **`huggingface-cli`** ([download guide](https://huggingface.co/docs/huggingface_hub/guides/download)); install the latest `huggingface_hub`.

Example for ChemBERTa:

```bash
MODEL_DIR="ChemBERTa-77M-MTR"
HF_PATH="DeepChem/ChemBERTa-77M-MTR"   # BioT5: QizhiPei/biot5-base | MolT5: laituan245/molt5-base
mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR"
huggingface-cli download "$HF_PATH" --local-dir ./
```

Place directories under `DeepFP_Prep/utils/assets/` (or your `DEEPFP_ASSETS_DIR`) as:

| Model     | Hugging Face repo | Typical local folder name   |
|-----------|-------------------|-----------------------------|
| ChemBERTa | `DeepChem/ChemBERTa-77M-MTR` | `ChemBERTa-77M-MTR/` |
| BioT5     | `QizhiPei/biot5-base`        | `BioT5/`             |
| MolT5     | `laituan245/molt5-base`      | `MolT5/`             |

Optional direct weight URLs (same files as above):

- [MolT5 `pytorch_model.bin`](https://huggingface.co/laituan245/molt5-base/resolve/main/pytorch_model.bin)
- [BioT5 `pytorch_model.bin`](https://huggingface.co/QizhiPei/biot5-base/resolve/main/pytorch_model.bin)
- [ChemBERTa `pytorch_model.bin`](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/resolve/main/pytorch_model.bin)

### SSL-based graph models (GraphMVP Google Drive)

These checkpoints come from the **GraphMVP** release area on **Google Drive** ([folder link](https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6?usp=sharing)), as described in MolRetrieval and the [GraphMVP README](https://github.com/chao1224/GraphMVP). Download and **rename** into `utils/assets/`:

| File on Drive (MolRetrieval names) | Save as (`utils/assets/…`) |
|-----------------------------------|----------------------------|
| `Motif.pth` (Grover)              | `Motif/model.pth`          |
| `AM.pth` (AttrMask)               | `AM/model.pth`             |
| `GPT_TNN.pth` (GPT-GNN)           | `GPT_GNN/model.pth`        |
| `GraphCL.pth`                     | `GraphCL/model.pth`        |

**GraphMVP:** download **`GraphMVP_complate_features_for_regression.zip`** (spelling per [MolRetrieval README](https://github.com/linhaowei1/MolRetrieval)), then copy  
`GraphMVP_complate_features_for_regression/GraphMVP/pretraining_model.pth` → **`GraphMVP/model.pth`**.

### MolCLR

Download [`model.pth` from the official MolCLR repo](https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth) → **`MolCLR/model.pth`**.

### Uni-Mol V1

Repo: [`dptech/Uni-Mol-Models`](https://huggingface.co/dptech/Uni-Mol-Models) (same as `utils/assets/UniMolV1/weighthub.py`).

| File | URL |
|------|-----|
| `mol_pre_no_h_220816.pt` | [resolve/main/mol_pre_no_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_no_h_220816.pt) |
| `mol_pre_all_h_220816.pt` (optional) | [resolve/main/mol_pre_all_h_220816.pt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt) |
| `mol.dict.txt` (if missing) | [resolve/main/mol.dict.txt](https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol.dict.txt) |

Place under **`UniMolV1/`**.

### Uni-Mol V2

Repo: [`dptech/Uni-Mol2`](https://huggingface.co/dptech/Uni-Mol2) — layout [`modelzoo/`](https://huggingface.co/dptech/Uni-Mol2/tree/main/modelzoo).

| Local path | Download URL |
|------------|--------------|
| `UniMolV2/84M/checkpoint.pt` | […/modelzoo/84M/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt) |
| `UniMolV2/310M/checkpoint.pt` | […/modelzoo/310M/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/310M/checkpoint.pt) |
| `UniMolV2/1.1B/checkpoint.pt` | […/modelzoo/1.1B/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/1.1B/checkpoint.pt) |

### Full experimental bundle (optional)

[Zenodo record](https://doi.org/10.5281/zenodo.18972759) — includes **`FP_set.tar.gz`** with aligned assets for paper reproduction.

---

## Outputs and splits

Use the repository **`cmd_fp`** conda environment ([`cmd_fp.yml`](../cmd_fp.yml) at repo root). Chunked outputs: `*_batch_*.pkl`. Convert or merge to `{task}_{train|valid|test}.pkl` for ArcMol `data_dir`.

For `feature_process.py`, set **`EMBEDDING_MODE`** (`"all"` vs `"allowed"`) and use **`--list-embeddings`** as needed. If anything from `requirements-deepfp-prep.txt` is missing in your env, install it with `pip`.

---

## Layout

```
DeepFP_Prep/
├── feature_process.py
├── embed.py
├── read_pkls.py
├── requirements-deepfp-prep.txt
├── models/
├── utils/
│   ├── env_utils.py
│   └── assets/
│       ├── WEIGHTS_DOWNLOAD_LIST.txt   # short English index (this page is canonical)
│       └── … tokenizer dirs + weight subfolders …
```

---

## Included vs not included

| Included | Not included |
|----------|----------------|
| Code + small tokenizer/config files | `.pth` / `.pt` / large `pytorch_model.bin` |
| This README | Raw CSVs / huge prebuilt pkls |
