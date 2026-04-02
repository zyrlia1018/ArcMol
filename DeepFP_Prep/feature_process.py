# smiles2feats.py  ——  在当前 no-valid 版本基础上，补充 RDKit 全量描述符写入
"""
将 CSV 转为按 split 分块的 pickle：每条记录含标准化 SMILES、可选 target、多种 embedding 向量、
可选字段 rdkit_descriptors（RDKit 全量标量描述符）。

================================================================================
【表征（embedding）总览】
================================================================================
名称由同目录 embed.py 里的 EmbeddingRegistry 注册；「本机当前能算哪些」以运行时的
Embedding.available() 为准（缺依赖的注册项不会出现在列表里，例如部分指纹需 Avalon/Pharm2D）。

embed.py 源码中已写死的注册名（供对照；实际可用 = 与 available 取交集）：

  • 描述符 / 指纹类
      RDKitDescriptors, Mordred（视环境）,
      RDKFingerprint, MACCSkeys, EStateFingerprint, MorganFingerprint,
      ECFP2, ECFP4, ECFP6, FCFP2, FCFP4, FCFP6,
      PatternFingerprint, LayeredFingerprint,
      AvalonFingerprint（视环境）, Pharm2D（视环境）

  • 预训练模型类
      MolT5, BioT5, ChemBERTa, MolCLR,
      AttrMask, GPT-GNN, GraphCL, GraphMVP, GROVER,
      UniMolV1, UniMolV2_84M, UniMolV2_310M, UniMolV2_1.1B

本脚本里 RDKFingerprint / MACCSkeys / EStateFingerprint 走轻量 RDKit 路径；其余名称通过
Embedding.get 计算。

================================================================================
【选哪些表征：模式】
================================================================================
在 __main__ 中设置 EMBEDDING_MODE，或调用 resolve_embedding_names(mode)：

  • "all"    — 计算 Embedding.available() 中的全部（最重、最全）。
  • "allowed" — 仅计算常量 ALLOWED_EMBEDDINGS 与 available 的交集（默认子集，见文件中部定义）。

查看「你这台机器」真实可用列表：

  • 命令行（在 DeepFP 目录下）:
        python feature_process.py --list-embeddings

  • 或代码:
        from feature_process import print_available_embeddings
        print_available_embeddings()
"""
from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Literal, FrozenSet
import argparse
import os, gc, shutil, warnings, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None

# ---- 安静化（需在导入模型/嵌入前）----
os.environ["OGB_DISABLE_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
from rdkit import RDLogger, DataStructs
RDLogger.DisableLog("rdApp.warning")
# -------------------------------------

# -------- SMILES 标准化（按你给的实现）--------
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem.MolStandardize import rdMolStandardize
def standardize_smiles(smiles: str,
                       basic_clean: bool=True,
                       clear_charge: bool=True,
                       clear_fragment: bool=True,
                       canonicalize_tautomer: bool=True,
                       isomeric: bool=False) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        if basic_clean:    mol = rdMolStandardize.Cleanup(mol)
        if clear_fragment: mol = rdMolStandardize.FragmentParent(mol)
        if clear_charge:   mol = rdMolStandardize.Uncharger().uncharge(mol)
        if canonicalize_tautomer:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None
# --------------------------------------------

# -------- RDKit 全量描述符（参考你的 data_prepare 脚本做法）--------
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def compute_rdkit_desc(smiles: str) -> Dict[str, float]:
    """枚举 RDKit 内置描述符；异常/NaN/Inf 置 0，返回 {name: float}。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        names = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        vals = calc.CalcDescriptors(mol)
        out = {}
        for k, v in zip(names, vals):
            try:
                x = float(v)
                if not np.isfinite(x):
                    x = 0.0
            except Exception:
                x = 0.0
            out[k] = x
        return out
    except Exception:
        return {}
# ------------------------------------------------------------------

_LIGHTWEIGHT_EMBEDDINGS = {"RDKFingerprint", "MACCSkeys", "EStateFingerprint"}

def _fp_to_np(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32, copy=False)

def _embed_batch_lightweight(name: str, smiles_batch: List[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"无效 SMILES: {smi}")
        if name == "RDKFingerprint":
            rows.append(_fp_to_np(Chem.RDKFingerprint(mol)))
        elif name == "MACCSkeys":
            rows.append(_fp_to_np(MACCSkeys.GenMACCSKeys(mol)))
        elif name == "EStateFingerprint":
            est, _ = Fingerprinter.FingerprintMol(mol)
            rows.append(np.asarray(est, dtype=np.float32))
        else:
            raise ValueError(f"不支持的轻量 embedding: {name}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)

def _build_embedder(name: str, batch_size: int):
    # 只有用到重模型 embedding 时才导入 embed.py，避免轻量模式强依赖深度学习栈。
    from embed import Embedding
    return Embedding.get(name, batch_size=batch_size)

def _normalize_split(df: pd.DataFrame, split_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    不做 train->valid 的拆分：
      - 若 split_col 为空或不存在：新建 '_split_tmp_'='all'
      - 若存在：小写化，原样使用
    """
    if not split_col or split_col not in df.columns:
        out = df.copy()
        out["_split_tmp_"] = "all"
        return out, "_split_tmp_"
    out = df.copy()
    out[split_col] = out[split_col].astype(str).str.lower()
    return out, split_col

def csv_to_pkls(csv_path: str,
                         out_dir: str,
                         *,
                         smiles_col: str = "smiles",
                         y_col: Optional[str] = "y",
                         target_prefix: str = "target_",
                         extra_cols: Optional[List[str]] = None,   # 附加保留列
                         split_col: Optional[str] = "split",       # None → 全部写到 'all'
                         embedding_names: List[str] = None,        # 要计算的 embeddings
                         batch_size: int = 256,
                         chunk_size: int = 20000,
                         include_rdkit: bool = True                # 是否写入 rdkit_descriptors
                         ) -> List[str]:
    """
    必写：'SMILES'（标准化后）。
    可选：target 列（当 y_col 存在时，键名为 f'{target_prefix}{y_col}'）。
    可写：extra_cols（若存在）、各 embedding、以及 rdkit_descriptors（可选）。
    """
    if not embedding_names:
        raise ValueError("embedding_names 不能为空")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp_memmap"); os.makedirs(tmp_dir, exist_ok=True)

    # 1) 读取 + 列检查
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns: raise ValueError(f"缺少列：{smiles_col}")
    has_target = bool(y_col) and (y_col in df.columns)
    if y_col and not has_target:
        print(f"[WARN] 未找到目标列 '{y_col}'，将只写入分子特征，不写 target。")
    target_key = f"{target_prefix}{y_col}" if (target_prefix is not None) else y_col

    # 2) 标准化 SMILES（失败剔除）
    raw = df[smiles_col].astype(str).tolist()
    smiles_std, keep = [], []
    for s in tqdm(raw, desc="Standardizing SMILES", ncols=100):
        st = standardize_smiles(s); smiles_std.append(st); keep.append(st is not None)
    keep = np.array(keep, bool)
    if (~keep).sum() > 0:
        print(f"[WARN] 标准化失败剔除 {int((~keep).sum())} 条")
    df = df.loc[keep].reset_index(drop=True)
    smiles_std = [x for x, k in zip(smiles_std, keep) if k]
    if len(df) == 0:
        raise ValueError("标准化后无有效 SMILES，无法继续。")
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    # 3) split 规范化
    df, split_col = _normalize_split(df, split_col)

    # 4) 逐个 embedding（内部 batch）→ memmap（避免堆内存/显存）
    N = len(smiles_std)
    mmap_info: Dict[str, Tuple[str, int]] = {}
    for name in embedding_names:
        emb = None

        # 先跑一小批确定维度
        end0 = min(batch_size, N)
        if name in _LIGHTWEIGHT_EMBEDDINGS:
            first = _embed_batch_lightweight(name, smiles_std[:end0])
        else:
            emb = _build_embedder(name, batch_size=batch_size)
            first = emb(smiles_std[:end0], batch_size=batch_size).numpy().astype(np.float32, copy=False)
        D = first.shape[1]

        mpath = os.path.join(tmp_dir, f"{name}.mmap")
        mmap = np.memmap(mpath, dtype=np.float32, mode="w+", shape=(N, D))
        mmap[:end0, :] = first

        for s in tqdm(range(end0, N, batch_size), ncols=100, desc=f"[{name}] featurizing"):
            e = min(s + batch_size, N)
            if name in _LIGHTWEIGHT_EMBEDDINGS:
                arr = _embed_batch_lightweight(name, smiles_std[s:e])
            else:
                arr = emb(smiles_std[s:e], batch_size=batch_size).numpy().astype(np.float32, copy=False)
            mmap[s:e, :] = arr
        mmap.flush(); del mmap
        mmap_info[name] = (mpath, D)

        # 清理
        try:
            del emb
            if torch is not None and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            gc.collect()
        except Exception: pass

    # 5) 按 split 分块写 pkl；此处顺便写入 RDKit 描述符（不占大内存）
    out_files: List[str] = []
    for sp in df[split_col].unique().tolist():
        sub = df.loc[df[split_col] == sp].copy()
        idxs = sub["row_id"].to_numpy(); n = len(idxs)
        if n == 0: continue

        # 只在需要时打开 mmap 读句柄
        mmaps: Dict[str, np.memmap] = {
            name: np.memmap(mpath, dtype=np.float32, mode="r", shape=(N, D))
            for name, (mpath, D) in mmap_info.items()
        }

        for bi in range(0, n, chunk_size):
            be = min(bi + chunk_size, n)
            rows = idxs[bi:be]
            pack: Dict[int, Dict] = {}

            sub_indexed = sub.set_index("row_id", drop=False)
            for rid in rows:
                rid = int(rid)
                rec = {
                    "SMILES": smiles_std[rid],
                }
                if has_target:
                    rec[target_key] = sub_indexed.at[rid, y_col]
                # 附加列（存在才写）
                if extra_cols:
                    for col in extra_cols:
                        if col in sub.columns:
                            rec[col] = sub_indexed.at[rid, col]

                # 写各 embedding
                for name, mm in mmaps.items():
                    rec[name] = mm[rid].copy()

                # 写 RDKit 描述符（可关）
                if include_rdkit:
                    rec["rdkit_descriptors"] = compute_rdkit_desc(smiles_std[rid])

                pack[rid] = rec

            out_path = os.path.join(out_dir, f"{sp}_batch_{bi}_{be}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
            out_files.append(out_path)
            print(f"[OK] {sp}: rows {bi}..{be-1} -> {out_path}")

        # 关闭 mmap
        for mm in mmaps.values(): del mm

    # 6) 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_files


# 指定子集：mode=="allowed" 时使用（与 Embedding.available() 取交集，未注册则 WARN 跳过）
ALLOWED_EMBEDDINGS: FrozenSet[str] = frozenset({
    "RDKFingerprint",
    "MACCSkeys",
    "EStateFingerprint",
    "MolT5",
    "BioT5",
    "AttrMask",
    "GPT-GNN",
    "GraphCL",
    "MolCLR",
    "GraphMVP",
    "GROVER",
    "UniMolV1",
    "UniMolV2_84M",
    "UniMolV2_1.1B",
})

EmbeddingMode = Literal["all", "allowed"]


def print_available_embeddings() -> None:
    """打印本机 Embedding.available() 排序列表（与 embed 实际可算的一致）。"""
    from embed import Embedding
    names = sorted(Embedding.available())
    print(f"共 {len(names)} 项（按名排序）：")
    for n in names:
        print(f"  {n}")


def all_registered_embedding_names() -> List[str]:
    """与 embed.py 当前可用注册一致的全部表征名（排序，便于复现）。"""
    from embed import Embedding
    return sorted(Embedding.available())


def resolve_embedding_names(mode: EmbeddingMode | str) -> List[str]:
    """
    - "all": 与 all_registered_embedding_names() 相同，即 Embedding.available() 全部。
    - "allowed": ALLOWED_EMBEDDINGS ∩ available()；不在本机的项会 WARN 并跳过。
    完整注册名对照见本模块顶部文档字符串；实时列表用 print_available_embeddings()。
    """
    m = (mode or "allowed").strip().lower()
    from embed import Embedding
    available = set(Embedding.available())
    if m == "all":
        names = sorted(available)
    elif m == "allowed":
        wanted = ALLOWED_EMBEDDINGS & available
        missing = ALLOWED_EMBEDDINGS - available
        if missing:
            print(
                f"[WARN] ALLOWED_EMBEDDINGS 中 {len(missing)} 项当前环境未注册，已跳过: "
                f"{sorted(missing)}"
            )
        names = sorted(wanted)
    else:
        raise ValueError(f"未知 embedding mode: {mode!r}，请用 'all' 或 'allowed'")
    if not names:
        raise RuntimeError("无可用 embedding 名称，请检查 mode 与 embed 环境。")
    return names


# ============ 可改动变量 & 示例 ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CSV→pkl 特征；用 --list-embeddings 查看本机全部可用表征名")
    ap.add_argument(
        "--list-embeddings",
        action="store_true",
        help="仅打印 Embedding.available() 列表后退出",
    )
    args = ap.parse_args()
    if args.list_embeddings:
        print_available_embeddings()
        raise SystemExit(0)

    # "all" = 全部已注册表征；"allowed" = 仅 ALLOWED_EMBEDDINGS 子集（见模块文档与下方常量）
    EMBEDDING_MODE: EmbeddingMode = "allowed"

    embedding_names = resolve_embedding_names(EMBEDDING_MODE)

    # 最小示例：只有 smiles（也可额外提供 y / split / 其它列）
    df_demo = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
    })
    demo_csv = "demo_input.csv"
    df_demo.to_csv(demo_csv, index=False)

    out_files = csv_to_pkls(
        csv_path=demo_csv,
        out_dir="./pkls_no_valid",
        smiles_col="smiles",
        y_col="y",                                    # 若列不存在会自动跳过 target
        extra_cols=[],                                # 想保留的列名
        split_col="split",                            # 若列不存在会自动写到 all
        embedding_names=embedding_names,
        batch_size=256,
        chunk_size=20000,
        include_rdkit=True,  # <—— 这里打开 RDKit 全量特征
    )
    print("[DONE]", out_files)


