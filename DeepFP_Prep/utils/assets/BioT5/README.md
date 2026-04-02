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