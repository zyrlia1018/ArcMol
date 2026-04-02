import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel
from utils.env_utils import CHEMBERTA_PATH  # 你原来的路径

def init_chemberta():
    # 1. 加载 tokenizer（保持不变）
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_PATH)

    # 2. 加载 config，但不建 pooler
    config = RobertaConfig.from_pretrained(CHEMBERTA_PATH)

    # 3. 加载模型，明确关掉 pooling layer
    model = RobertaModel.from_pretrained(
        CHEMBERTA_PATH,
        config=config,
        add_pooling_layer=False,   # ← 关键，避免那个 warning
    )

    model.cuda()
    model.eval()
    return tokenizer, model

@torch.no_grad()
def chemberta_embedding(smiles_block, tokenizer, model):
    # 跟你原来一样：一批 SMILES → CLS
    batch = tokenizer(
        smiles_block,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )
    batch = {k: v.cuda() for k, v in batch.items()}
    out = model(**batch).last_hidden_state[:, 0]   # 直接取 CLS
    out = out.detach().cpu()
    torch.cuda.empty_cache()
    return out
