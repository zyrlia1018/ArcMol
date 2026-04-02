# models/UniMolV2/utils.py
import os
import shutil
import torch

from utils.env_utils import (
    UNIMOL2_ROOT,
    UNIMOL2_84M_PATH,
    UNIMOL2_164M_PATH,
    UNIMOL2_310M_PATH,
    UNIMOL2_570M_PATH,
    UNIMOL2_1_1B_PATH,
)

try:
    from unimol_tools import UniMolRepr
except Exception:
    UniMolRepr = None

UNI2_LOCAL = {
    "UniMol2_84M":  (UNIMOL2_84M_PATH,  "84m"),
    "UniMol2_164M": (UNIMOL2_164M_PATH, "164m"),
    "UniMol2_310M": (UNIMOL2_310M_PATH, "310m"),
    "UniMol2_570M": (UNIMOL2_570M_PATH, "570m"),
    "UniMol2_1.1B": (UNIMOL2_1_1B_PATH, "1.1B"),
}

SIZE_DIR_MAP = {
    "84m": "84M",
    "164m": "164M",
    "310m": "310M",
    "570m": "570M",
    "1.1B": "1.1B",
}


def _ensure_local_weight_for_unimol_tools(local_ckpt: str, model_size: str):
    size_dir = SIZE_DIR_MAP[model_size]
    target_dir = os.path.join(UNIMOL2_ROOT, "modelzoo", size_dir)
    target_ckpt = os.path.join(target_dir, "checkpoint.pt")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_ckpt):
        return

    if os.path.exists(local_ckpt):
        try:
            os.symlink(local_ckpt, target_ckpt)
        except OSError:
            shutil.copy2(local_ckpt, target_ckpt)


def _silence_unimol_logs():
    import logging
    noisy = [
        "unimol_tools",
        "unimol_tools.models.unimolv2",
        "unimol_tools.data.conformer",
        "unimol_tools.tasks.trainer",
        "numba",
        "numba.core",
        "numba.core.ssa",
    ]
    for name in noisy:
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
        # 清理它加的 handler，防止继续往 stdout 打
        while lg.handlers:
            lg.removeHandler(lg.handlers[0])


def init_unimol2(tag: str = "UniMol2_84M", device: str = "cuda"):
    if UniMolRepr is None:
        raise RuntimeError("unimol_tools is not installed.")

    if tag not in UNI2_LOCAL:
        raise ValueError(f"Unknown UniMol2 tag: {tag}")

    local_ckpt, model_size = UNI2_LOCAL[tag]

    # 1) 重定向两个 WEIGHT_DIR
    import importlib
    wh = importlib.import_module("unimol_tools.weights.weighthub")
    wh.WEIGHT_DIR = UNIMOL2_ROOT
    unimolv2_mod = importlib.import_module("unimol_tools.models.unimolv2")
    unimolv2_mod.WEIGHT_DIR = UNIMOL2_ROOT

    # 2) 确保本地有它要看的 checkpoint
    _ensure_local_weight_for_unimol_tools(local_ckpt, model_size)

    # 3) 初始化模型
    if os.path.exists(local_ckpt):
        clf = UniMolRepr(
            data_type="molecule",
            remove_hs=True,
            model_name="unimolv2",
            model_size=model_size,
            ckpt_path=local_ckpt,
            device=device,
        )
    else:
        clf = UniMolRepr(
            data_type="molecule",
            remove_hs=True,
            model_name="unimolv2",
            model_size=model_size,
            device=device,
        )

    # 4) 初始化完再静音（一定要放这里）
    _silence_unimol_logs()

    # 5) dim
    if model_size in ("84m", "164m", "310m"):
        dim = 768
    elif model_size == "570m":
        dim = 1024
    else:
        dim = 1536

    return clf, dim


@torch.no_grad()
def unimol2_embedding(smiles, model):
    out = model.get_repr(smiles, return_atomic_reprs=False)
    if isinstance(out, dict) and "cls_repr" in out:
        cls = out["cls_repr"]
        if isinstance(cls, list):
            import numpy as np
            arr = np.stack(cls, axis=0)
            return torch.from_numpy(arr).float()
        import numpy as np
        if isinstance(cls, np.ndarray):
            return torch.from_numpy(cls).float()
    if isinstance(out, torch.Tensor):
        return out.detach().cpu()
    raise RuntimeError("Unexpected UniMol2 output format.")
