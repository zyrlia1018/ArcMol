# `utils/assets` — pre-trained checkpoints (layout)

`DeepFP_Prep` resolves all backbone paths from **`env_utils.py`**: by default `DeepFP_Prep/utils/assets`, or override with **`DEEPFP_ASSETS_DIR`**.

GitHub carries **small files** (tokenizers, configs, `mol.dict.txt`, …). **Large weights** (`.pth`, `.pt`, `pytorch_model.bin`) are omitted from Git; add them locally, unpack a Zenodo bundle, or point `DEEPFP_ASSETS_DIR` at your mirror.

| Directory | Role |
|-----------|------|
| `ChemBERTa-77M-MTR/` | ChemBERTa tokenizer + **`pytorch_model.bin`** — see folder `README.md` |
| `BioT5/` | Tokenizer + **`pytorch_model.bin`** — see `README.md` |
| `MolT5/` | Tokenizer + **`pytorch_model.bin`** — see `README.md` |
| `UniMolV1/` | **`mol_pre_no_h_220816.pt`** (and optional `mol_pre_all_h_220816.pt`) — see `README.md` |
| `UniMolV2/` | Subfolders `84M/`, `310M/`, `1.1B/` with **`checkpoint.pt`** — see `README.md` |
| `MolCLR/` | **`model.pth`** — see `README.md` |
| `Motif/` | GROVER-style **`model.pth`** (from Drive `Motif.pth`) — see `README.md` |
| `AM/` | AttrMask **`model.pth`** (from Drive `AM.pth`) — see `README.md` |
| `GPT_GNN/` | **`model.pth`** (from Drive `GPT_TNN.pth`) — see `README.md` |
| `GraphCL/` | **`model.pth`** (from Drive `GraphCL.pth`) — see `README.md` |
| `GraphMVP/` | **`model.pth`** (from GraphMVP zip `pretraining_model.pth`) — see `README.md` |

**Machine-readable index:** [`WEIGHTS_DOWNLOAD_LIST.txt`](WEIGHTS_DOWNLOAD_LIST.txt)  

**Narrative + tables:** [`../README.md`](../README.md) (repository root DeepFP_Prep section)
