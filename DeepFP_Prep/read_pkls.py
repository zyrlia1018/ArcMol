# read_pkls_v2.py
from __future__ import annotations
import os, argparse, pickle
from glob import glob
from typing import Any, Dict, List, Tuple, Iterable
import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

TARGET_PREFIX = "target_"
# 这些键一律视为“元信息”，不是嵌入；按需增减
DEFAULT_META_KEYS = {
    "SMILES", "split", "cliff_mol", "note", "rdkit_descriptors",
}

def list_pkls(pkl_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(pkl_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"没找到 pkl：{pkl_dir}")
    return paths

def load_pack(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)

def _to_numpy_1d(x: Any) -> np.ndarray | None:
    """尽量把各种容器转成一维 float32 向量；失败返回 None。"""
    try:
        if HAS_TORCH and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple)):
            x = np.asarray(x)
        elif isinstance(x, np.ndarray):
            pass
        else:
            return None
        x = np.asarray(x)
        # 压平到 1D（很多库会给 (D,) 或 (1,D)）
        if x.ndim > 1:
            x = x.reshape(-1)
        if x.dtype.kind not in ("f", "i", "u"):
            # 非数值类型直接放弃
            return None
        return x.astype(np.float32, copy=False)
    except Exception:
        return None

def detect_embedding_keys(rec: Dict[str, Any],
                          meta_keys: Iterable[str]) -> List[str]:
    """从样本字典里筛出可能的嵌入键。"""
    embs = []
    for k, v in rec.items():
        if k in meta_keys or k.startswith(TARGET_PREFIX):
            continue
        arr = _to_numpy_1d(v)
        if arr is not None and arr.ndim == 1 and arr.size > 0:
            embs.append(k)
    return sorted(embs)

def peek_pkl(path: str, n_show: int = 2,
             meta_keys: Iterable[str] = DEFAULT_META_KEYS) -> Tuple[List[str], Dict[str,int]]:
    pack = load_pack(path)
    keys = sorted(pack.keys())
    print(f"\n=== PKL: {os.path.basename(path)} ===")
    print(f"样本数: {len(pack)} | row_id 最小/最大: {keys[0]} / {keys[-1]}")
    if not keys:
        return [], {}

    first = pack[keys[0]]
    emb_names = detect_embedding_keys(first, meta_keys)
    dims = {}
    for name in emb_names:
        arr = _to_numpy_1d(first[name])
        dims[name] = int(arr.shape[0]) if arr is not None else -1
    print(f"嵌入种类({len(emb_names)}): {emb_names}")
    print(f"各维度: {dims}")

    show_ids = keys[:min(n_show, len(keys))]
    for rid in show_ids:
        rec = pack[rid]
        targets = {k: rec[k] for k in rec if k.startswith(TARGET_PREFIX)}
        extras  = {k: rec[k] for k in rec if (k in meta_keys and k != "SMILES")}
        print(f"\n[row_id={rid}]")
        print("  SMILES:", rec.get("SMILES"))
        if targets:
            print("  Targets:", targets)
        if extras:
            print("  Extra  :", extras)
        print("  Embeds :", {e: _to_numpy_1d(rec[e]).shape for e in emb_names})
    return emb_names, dims

def stack_embedding_matrix(pkl_dir: str, emb_name: str) -> np.ndarray:
    """把某个嵌入在整个目录内按 row_id 叠成 [N, D] 矩阵（按文件与 row_id 递增拼接）。"""
    mats = []
    for path in list_pkls(pkl_dir):
        pack = load_pack(path)
        row_ids = sorted(pack.keys())
        rows = []
        for rid in row_ids:
            arr = _to_numpy_1d(pack[rid].get(emb_name, None))
            if arr is None:
                raise ValueError(f"{os.path.basename(path)} row_id={rid} 缺少嵌入 {emb_name}")
            rows.append(arr)
        mats.append(np.stack(rows, axis=0))
    return np.concatenate(mats, axis=0) if mats else np.zeros((0,0), dtype=np.float32)

def load_meta_df(pkl_dir: str) -> pd.DataFrame:
    """只汇总元信息与 target_* 到 DataFrame（不展开嵌入向量）。"""
    rows = []
    for path in list_pkls(pkl_dir):
        split_guess = os.path.basename(path).split("_batch_")[0]
        pack = load_pack(path)
        for rid, rec in pack.items():
            row = {"row_id": rid, "SMILES": rec.get("SMILES"), "split_file": split_guess}
            for k, v in rec.items():
                if k.startswith(TARGET_PREFIX):
                    row[k] = v
            for k in DEFAULT_META_KEYS:
                if k in ("SMILES",):  # 避免重复
                    continue
                if k in rec:
                    row[k] = rec[k]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_file","row_id"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="存放 pkl 的目录")
    ap.add_argument("--peek", type=int, default=2, help="每个 pkl 预览多少条")
    ap.add_argument("--meta-csv", default="", help="把元信息导出成 CSV（不含嵌入）")
    ap.add_argument("--stack", default="", help="把指定嵌入名整合为一个 .npy（按文件和 row_id 顺序）")
    ap.add_argument("--stack-out", default="", help="stack 输出路径（.npy），不填则用 <emb>.npy")
    args = ap.parse_args()

    paths = list_pkls(args.dir)
    print(f"共发现 {len(paths)} 个 pkl：")
    for p in paths:
        print(" -", os.path.basename(p))

    last_embs = None
    for p in paths:
        emb_names, _ = peek_pkl(p, n_show=args.peek, meta_keys=DEFAULT_META_KEYS)
        last_embs = emb_names

    if args.meta_csv:
        df = load_meta_df(args.dir)
        df.to_csv(args.meta_csv, index=False)
        print(f"\n[OK] 导出元信息 -> {args.meta_csv} (行数={len(df)})")

    if args.stack:
        mat = stack_embedding_matrix(args.dir, args.stack)
        out = args.stack_out or f"{args.stack}.npy"
        np.save(out, mat)
        print(f"[OK] 保存 {args.stack} 到 {out} | shape={mat.shape}")

if __name__ == "__main__":
    main()


