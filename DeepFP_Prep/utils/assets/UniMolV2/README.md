# Uni-Mol V2 — checkpoints for DeepFP_Prep

## Layout

Place Hugging Face **`checkpoint.pt`** files under size-specific subfolders (see `../../env_utils.py`):

```text
UniMolV2/84M/checkpoint.pt
UniMolV2/310M/checkpoint.pt
UniMolV2/1.1B/checkpoint.pt
```

(Optional sizes such as `164M` / `570M` / `580M` are supported by code if present.)

## Download

Hub: [dptech/Uni-Mol2 — modelzoo](https://huggingface.co/dptech/Uni-Mol2/tree/main/modelzoo)

| Local path | Direct link |
|------------|-------------|
| `84M/checkpoint.pt` | […/84M/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt) |
| `310M/checkpoint.pt` | […/310M/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/310M/checkpoint.pt) |
| `1.1B/checkpoint.pt` | […/1.1B/checkpoint.pt](https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/1.1B/checkpoint.pt) |

Example:

```bash
mkdir -p UniMolV2/84M UniMolV2/310M UniMolV2/1.1B
wget -O UniMolV2/84M/checkpoint.pt \
  https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt
# repeat for other sizes as needed
```

## Reference

- Index: `../WEIGHTS_DOWNLOAD_LIST.txt`
