---
license: apache-2.0
---
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
