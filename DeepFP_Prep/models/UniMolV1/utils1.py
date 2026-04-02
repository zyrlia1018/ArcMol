# models/UniMolV1/utils.py
import torch
from typing import List, Tuple

# 有就用，没有就让 unimol_tools 用它自己的
try:
    from utils.env_utils import UNIMOL1_DICT_PATH
except Exception:
    UNIMOL1_DICT_PATH = None

try:
    from unimol_tools import UniMolRepr
except Exception:
    UniMolRepr = None


def _silence_unimol1_logs():
    import logging
    for name in [
        "unimol_tools",
        "unimol_tools.models.unimol",
        "unimol_tools.weights.weighthub",
        "numba",
    ]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
        while lg.handlers:
            lg.removeHandler(lg.handlers[0])


def init_unimol1(tag: str = "UniMol1_noH", device: str = "cuda") -> Tuple[object, int]:
    """
    官方流程：
      - 不看本地 pt
      - 不建 modelzoo
      - 交给 unimol_tools 自己下载
    tag:
      - "UniMol1_noH"  -> remove_hs=True
      - "UniMol1_allH" -> remove_hs=False
    """
    if UniMolRepr is None:
        raise RuntimeError("unimol_tools is not installed.")

    if tag == "UniMol1_noH":
        remove_hs = True
    elif tag == "UniMol1_allH":
        remove_hs = False
    else:
        raise ValueError(f"Unknown UniMolV1 tag: {tag}")

    kwargs = dict(
        data_type="molecule",
        remove_hs=remove_hs,
        model_name="unimolv1",   # v1 官方名
        device=device,
    )
    if UNIMOL1_DICT_PATH:
        kwargs["dict_path"] = UNIMOL1_DICT_PATH

    model = UniMolRepr(**kwargs)
    _silence_unimol1_logs()

    # v1 默认 512
    return model, 512


@torch.no_grad()
def unimol1_embedding(smiles: List[str], model) -> torch.Tensor:
    """
    输入: list[str] SMILES
    输出: torch.FloatTensor [B, 512] (CPU)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    out = model.get_repr(smiles, return_atomic_reprs=False)

    # 1) dict 格式
    if isinstance(out, dict):
        if "cls_repr" in out:
            x = out["cls_repr"]
        elif "molecule_repr" in out:
            x = out["molecule_repr"]
        else:
            raise RuntimeError("UniMolV1 output has no 'cls_repr' or 'molecule_repr'.")

        if isinstance(x, list):
            import numpy as np
            arr = np.stack(x, axis=0)
            return torch.from_numpy(arr).float()
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x.detach().cpu().float()
        import numpy as np
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t

    # 2) tensor
    if isinstance(out, torch.Tensor):
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.detach().cpu().float()

    # 3) numpy
    import numpy as np
    if isinstance(out, np.ndarray):
        t = torch.from_numpy(out).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    raise RuntimeError(f"Unexpected UniMolV1 output type: {type(out)}")
