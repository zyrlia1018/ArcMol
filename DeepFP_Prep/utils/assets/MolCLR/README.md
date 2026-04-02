# MolCLR — checkpoint for DeepFP_Prep

## Expected file

Save the pretrained GIN checkpoint as:

```text
MolCLR/model.pth
```

(i.e. this file should be named **`model.pth`** inside this folder.)

## Download

- **URL:** [model.pth (official MolCLR repo)](https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth)

Example:

```bash
mkdir -p MolCLR
wget -O MolCLR/model.pth \
  https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/checkpoints/model.pth
```

## Reference

- Code: [MolCLR](https://github.com/yuyangw/MolCLR)
- Path: `MOLCLR_PATH` in `../../env_utils.py`
- Index: `../WEIGHTS_DOWNLOAD_LIST.txt`
