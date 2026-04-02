---
license: apache-2.0
---
## Checkpoints for DeepFP_Prep

This directory usually contains **tokenizer and config** files only. For embedding inference you also need **`pytorch_model.bin`** here.

- **Recommended:** `huggingface-cli download laituan245/molt5-base --local-dir ./`
- **Direct file:** [pytorch_model.bin](https://huggingface.co/laituan245/molt5-base/resolve/main/pytorch_model.bin)

See `../WEIGHTS_DOWNLOAD_LIST.txt` and `../../env_utils.py`.

## Example Usage
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base')
```

## Paper

For more information, please take a look at our paper.

Paper: [Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)

Authors: *Carl Edwards\*, Tuan Lai\*, Kevin Ros, Garrett Honke, Heng Ji* 
