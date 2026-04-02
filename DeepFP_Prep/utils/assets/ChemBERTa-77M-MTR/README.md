# ChemBERTa-77M-MTR — checkpoints for DeepFP_Prep

This folder typically holds **tokenizer and config** files in Git. **Model weights** are large and must be added locally (or use Zenodo / `DEEPFP_ASSETS_DIR`).

## Download weights

Place **`pytorch_model.bin`** in this directory (same folder as `config.json`).

- **Recommended:** from the repository root, after `conda activate cmd_fp`:

  ```bash
  cd DeepFP_Prep/utils/assets/ChemBERTa-77M-MTR
  huggingface-cli download DeepChem/ChemBERTa-77M-MTR --local-dir ./
  ```

- **Direct file:** [pytorch_model.bin](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/resolve/main/pytorch_model.bin)

## Reference

- Hugging Face: [DeepChem/ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
- Path constant: `CHEMBERTA_PATH` in `../../env_utils.py`
- Full list: `../WEIGHTS_DOWNLOAD_LIST.txt`
