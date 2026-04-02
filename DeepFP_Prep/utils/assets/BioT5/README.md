---
license: mit
datasets:
- QizhiPei/BioT5_finetune_dataset
language:
- en
tags:
- biology
- chemistry
---
## Checkpoints for DeepFP_Prep

This directory usually contains **tokenizer and config** files only. For `embed.py` / `feature_process.py` you also need **`pytorch_model.bin`** in this same folder.

- **Recommended:** `huggingface-cli download QizhiPei/biot5-base --local-dir ./` (run from this directory; requires [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/download)).
- **Direct file:** [pytorch_model.bin](https://huggingface.co/QizhiPei/biot5-base/resolve/main/pytorch_model.bin)

Paths are defined in `../../env_utils.py`. Full index: `../WEIGHTS_DOWNLOAD_LIST.txt`.

## Example Usage
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-base')
```

## References
For more information, please refer to our paper and GitHub repository.

Paper: [BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations](https://arxiv.org/abs/2310.07276)

GitHub: [BioT5](https://github.com/QizhiPei/BioT5)

Authors: *Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, and Rui Yan*