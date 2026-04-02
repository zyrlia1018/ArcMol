# models/UniMolV1/utils.py
import os
import torch
from typing import List, Tuple

from utils.env_utils import (
    UNIMOL1_ROOT,
    UNIMOL1_DICT_PATH,
)

from unimol_tools import UniMolRepr
import unimol_tools.weights.weighthub as wh
import unimol_tools.models.unimol as unimol_v1_mod


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
    # ① 把“默认下载/加载目录”改成你自己的
    wh.WEIGHT_DIR = UNIMOL1_ROOT
    unimol_v1_mod.WEIGHT_DIR = UNIMOL1_ROOT

    # ② 选去氢 / 不去氢
    if tag == "UniMol1_noH":
        remove_hs = True
    elif tag == "UniMol1_allH":
        remove_hs = False
    else:
        raise ValueError(f"Unknown UniMolV1 tag: {tag}")

    # ③ 按官方方式建
    kwargs = dict(
        data_type="molecule",
        remove_hs=remove_hs,
        model_name="unimolv1",     # V1 实际用这个名
        device=device,
    )
    if UNIMOL1_DICT_PATH and os.path.exists(UNIMOL1_DICT_PATH):
        kwargs["dict_path"] = UNIMOL1_DICT_PATH

    model = UniMolRepr(**kwargs)
    _silence_unimol1_logs()
    return model, 512


@torch.no_grad()
def unimol1_embedding(smiles: List[str], model) -> torch.Tensor:
    if isinstance(smiles, str):
        smiles = [smiles]
    out = model.get_repr(smiles, return_atomic_reprs=False)

    if isinstance(out, dict):
        x = out.get("cls_repr") or out.get("molecule_repr")
        if isinstance(x, list):
            import numpy as np
            arr = np.stack(x, axis=0)
            return torch.from_numpy(arr).float()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float()
        import numpy as np
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t

    if isinstance(out, torch.Tensor):
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.detach().cpu().float()

    import numpy as np
    if isinstance(out, np.ndarray):
        t = torch.from_numpy(out).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    raise RuntimeError(f"Unexpected UniMolV1 output: {type(out)}")
