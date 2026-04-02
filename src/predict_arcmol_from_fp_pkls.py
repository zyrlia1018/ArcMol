# -*- coding: utf-8 -*-
"""
predict_arcmol_from_fp_pkls.py — 无标签推理（仅表征 pkl）

与 test_only_arcmol.py 区别：
  - 不需要 {task}_test.pkl 单文件；改为读取目录下 DeepFP 分块 *.pkl。
  - 无真实标签：不写 AUC/RMSE 等「测试指标」，只输出预测。
  - 默认模式：扫描 checkpoints/ 下全部任务（约 73 个），**只写一个**宽表 CSV（每列一个 endpoint）。
  - 仅调试单任务时用 --mode single（会单独写一个任务的 CSV）。

默认输出：`预测/admet_all_endpoints_preds.csv`（可用 --out_dir / --out_csv 改）。
不修改 test_only_arcmol.py；仅复用其 extract_extra_attributes。
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from main_arcmol_mcc_r2 import (
    set_seed,
    extract_rdkit_and_target,
    extract_selected_embedding,
    create_loader,
    TaskAwareDescriptorPooling,
    ArcMolModel,
)
from attention_pooling_fusion import AttentionPoolingFusion
from test_only_arcmol import extract_extra_attributes


# ---------- pkl 合并（DeepFP 输出）----------
def load_merged_fp_pkls(pkl_dir: str) -> Dict[int, Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"目录下无 .pkl: {pkl_dir}")
    merged: Dict[int, Dict[str, Any]] = {}
    i = 0
    for path in paths:
        with open(path, "rb") as f:
            pack = pickle.load(f)
        if not isinstance(pack, dict):
            raise TypeError(f"{path} 顶层应为 dict")
        for rid in sorted(pack.keys()):
            rec = dict(pack[rid])
            rec["_row_idx"] = i
            merged[i] = rec
            i += 1
    return merged


def load_merged_fp_pkl_file(path: str) -> Dict[int, Dict[str, Any]]:
    """
    读取单个已合并的 DeepFP 输出 pkl：顶层 dict，键为行号（int 或可排序），值为每条记录。
    与 load_merged_fp_pkls 合并后的结构一致，并写入 _row_idx。
    """
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        pack = pickle.load(f)
    if not isinstance(pack, dict):
        raise TypeError(f"{path} 顶层应为 dict")

    def _sort_key(k):
        if isinstance(k, int):
            return (0, k)
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    merged: Dict[int, Dict[str, Any]] = {}
    for i, rid in enumerate(sorted(pack.keys(), key=_sort_key)):
        rec = dict(pack[rid])
        rec["_row_idx"] = i
        merged[i] = rec
    return merged


def filter_checkpoint_jobs(
    jobs: List[Tuple[str, str, str]],
    exclude_dir_prefixes: Tuple[str, ...],
) -> List[Tuple[str, str, str]]:
    if not exclude_dir_prefixes:
        return jobs
    out = []
    for folder_name, bp, cp in jobs:
        if any(folder_name.startswith(pref) for pref in exclude_dir_prefixes):
            continue
        out.append((folder_name, bp, cp))
    return out


def run_all_checkpoints_wide(
    merged: Dict[int, Dict[str, Any]],
    checkpoints_root: str,
    out_csv: str,
    extra_attrs: List[str] | None = None,
    batch_size: int = 128,
    device: torch.device | None = None,
    exclude_dir_prefixes: Tuple[str, ...] = ("benchmark_",),
) -> pd.DataFrame:
    """
    扫描 checkpoints_root 下成对的 *.bundle.pt + 同名 .pth，对 merged 无标签数据逐任务推理，写宽表 CSV。
    供 notebook / 脚本复用。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if extra_attrs is None:
        extra_attrs = ["SMILES"]

    ck_root = Path(checkpoints_root)
    if not ck_root.is_dir():
        raise FileNotFoundError(f"checkpoints 目录不存在: {ck_root.resolve()}")

    jobs = discover_checkpoint_jobs(str(ck_root))
    jobs = filter_checkpoint_jobs(jobs, exclude_dir_prefixes)
    if not jobs:
        raise RuntimeError(
            f"未在 {ck_root} 下发现可用的 .bundle.pt+.pth 对（或全部被 exclude 过滤）"
        )
    print(
        f"[all_checkpoints] jobs={len(jobs)}  exclude_prefixes={exclude_dir_prefixes or 'none'}"
    )

    n_total = len(merged)
    meta = extract_extra_attributes(merged, extra_attrs)

    df = pd.DataFrame()
    for a in extra_attrs:
        df[a] = meta.get(a, [None] * n_total)

    ok, fail = 0, 0
    for folder_name, bp, cp in jobs:
        col_base = folder_name.replace("-", "_")
        try:
            preds, labels, slug, task_type, thr = run_one_bundle(
                bp, cp, merged, n_total, device, extra_attrs, batch_size
            )
            if task_type == "cls":
                df[f"{col_base}_pred_prob"] = preds
                if labels is not None:
                    df[f"{col_base}_pred_label"] = labels
            else:
                df[f"{col_base}_pred"] = preds
            ok += 1
            print(f"[OK] {folder_name} ({slug})  {task_type}")
        except Exception as e:
            fail += 1
            print(f"[SKIP] {folder_name}: {e}")

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"[Saved] {os.path.abspath(out_csv)}  cols={len(df.columns)}  ok={ok}  skip={fail}"
    )
    return df


# ---------- 无标签过滤与张量构造（与训练侧字段一致）----------
def _safe_flatten_1d(x):
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.dim() > 1:
            t = t.view(-1)
        return t.cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.reshape(-1)).cpu()
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.float32).view(-1).cpu()
    return torch.tensor([float(x)], dtype=torch.float32).view(-1).cpu()


def _bad(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        x = _safe_flatten_1d(x)
    if x.numel() == 0:
        return True
    if torch.isnan(x).any() or torch.isinf(x).any():
        return True
    if torch.all(x == 0):
        return True
    return False


def filter_for_inference(data_dict: Dict[int, Any], embed_types: List[str]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for k, rec in data_dict.items():
        if not isinstance(rec, dict):
            continue
        rd = rec.get("rdkit_descriptors")
        if not isinstance(rd, dict):
            continue
        ok = True
        for et in embed_types:
            arr = rec.get(et, None)
            if arr is None:
                ok = False
                break
            t = _safe_flatten_1d(arr)
            if _bad(t):
                ok = False
                break
        if ok:
            out[k] = rec
    return {k: out[k] for k in sorted(out.keys())}


def inject_dummy_target(
    data_dict: Dict[int, Dict[str, Any]],
    target_name: str,
    task_type: str,
) -> Dict[int, Dict[str, Any]]:
    out = {}
    for k, rec in data_dict.items():
        r = dict(rec)
        if task_type == "cls":
            r[target_name] = 0
        else:
            r[target_name] = 0.0
        out[k] = r
    return out


@torch.no_grad()
def forward_predictions(
    model,
    loader,
    device: torch.device,
    task_type: str,
    temperature: float | None,
    mc_passes: int,
    y_std_enable: bool,
    y_mu: float,
    y_sigma: float,
) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    for (emb_dict, d), _y in loader:
        emb_dict = {k: v.to(device) for k, v in emb_dict.items()}
        d = d.to(device)
        out_sum = 0.0
        for _ in range(mc_passes):
            if task_type == "cls":
                _ls, lt, _, _ = model.forward_logits((emb_dict, d), y=None, margin_scale=1.0)
                logits = lt
                if temperature is not None:
                    logits = logits / temperature
                out_sum = out_sum + logits
            else:
                _zs, zt, _, _ = model.forward_logits((emb_dict, d), y=None, margin_scale=1.0)
                out_sum = out_sum + zt
        out = out_sum / float(mc_passes)
        if task_type == "cls":
            prob = torch.softmax(out, dim=1)[:, 1]
            chunks.append(prob.detach().cpu().numpy())
        else:
            chunks.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([])


def build_model_from_bundle(bundle: dict, desc_te: torch.Tensor, device: torch.device):
    task_type = bundle["task"]["task_type"]
    fusion_embed_types = bundle["fusion_embed_types"]
    model_hparams = bundle["model_hparams"]

    desc_module = TaskAwareDescriptorPooling(
        in_dim=desc_te.shape[1], h=128, out_dim=64, drop=0.1
    ).to(device)
    fusion_module = AttentionPoolingFusion(
        used_embedding_types=fusion_embed_types,
        l_output_dim=model_hparams["fusion_out_dim"],
        hidden_dim=model_hparams["fusion_hidden_dim"],
        dropout_prob=model_hparams["fusion_dropout"],
        comp_mode=model_hparams["comp_mode"],
        cka_gamma=model_hparams["cka_gamma"],
        task_gate=model_hparams["task_gate"],
        task_ctx_dim=model_hparams["task_ctx_dim"],
        comp_scale=model_hparams["comp_scale"],
        top_k=model_hparams["moe_topk"],
        sparse_lambda=model_hparams["moe_sparse_lambda"],
    ).to(device)

    in_dim = model_hparams["fusion_out_dim"] + 64
    num_classes = 2 if task_type == "cls" else 1
    model = ArcMolModel(
        fusion_module,
        desc_module,
        in_dim=in_dim,
        task_type=task_type,
        num_classes=num_classes,
        task_ctx_dim=model_hparams["task_ctx_dim"],
        use_task_ctx=bool(model_hparams["use_task_ctx"]),
        margin=model_hparams["margin"],
        scale=model_hparams["scale"],
        head_hidden=model_hparams["head_hidden"],
        head_dropout=model_hparams["head_dropout"],
        proxy_dropout=model_hparams["proxy_dropout"],
        arc_reg_use=bool(model_hparams["arc_reg_use"]),
        arc_reg_nbins=model_hparams["arc_reg_nbins"],
        arc_reg_margin=model_hparams["arc_reg_margin"],
        arc_reg_scale=model_hparams["arc_reg_scale"],
        arc_reg_soft_sigma=model_hparams["arc_reg_soft_sigma"],
    ).to(device)
    return model, task_type


def run_one_bundle(
    bundle_path: str,
    ckpt_path: str,
    merged_raw: Dict[int, Dict[str, Any]],
    n_total: int,
    device: torch.device,
    extra_attrs: List[str],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray | None, str, str, float | None]:
    """
    返回 (pred_prob 或 pred_reg, pred_label 仅 cls 否则 None, task_slug, task_type, thr)
    pred 长度 n_total，无效行为 nan
    """
    bundle = torch.load(bundle_path, map_location="cpu")
    assert int(bundle.get("version", 1)) >= 1
    set_seed(int(bundle.get("seed", 24)))

    task_info = bundle["task"]
    task_type = task_info["task_type"]
    target_name = task_info["target_name"]
    fusion_embed_types = bundle["fusion_embed_types"]
    rdkit_meta = bundle["rdkit"]
    attributes = rdkit_meta["attribute_names"]
    topk = rdkit_meta["topk_idx"]
    scaler = rdkit_meta["scaler"]
    label_std = bundle.get("label_std", {"enable": False, "mu": 0.0, "sigma": 1.0})
    y_std_enable = bool(label_std.get("enable", False)) and (task_type == "reg")
    y_mu = float(label_std.get("mu", 0.0))
    y_sigma = float(label_std.get("sigma", 1.0))
    mc_passes = int(bundle.get("mc_passes", 12))
    calib = bundle.get("calibration", None)

    thr = float(calib.get("threshold", 0.5)) if calib and task_type == "cls" else None
    T = float(calib.get("temperature", 1.0)) if calib and task_type == "cls" else None

    filtered = filter_for_inference(merged_raw, fusion_embed_types)
    if len(filtered) == 0:
        raise RuntimeError("过滤后无样本（检查 embedding / rdkit_descriptors）")

    data_dict = inject_dummy_target(filtered, target_name, task_type)

    X, _y = extract_rdkit_and_target(data_dict, attributes, target=target_name, task_type=task_type)
    X_sel = X[:, topk]
    X_sel = scaler.transform(X_sel)
    X_sel = np.clip(X_sel, -1e6, 1e6)
    desc_te = torch.tensor(X_sel, dtype=torch.float32)

    emb_te, y_te = extract_selected_embedding(
        data_dict, fusion_embed_types, target=target_name, task_type=task_type
    )
    if task_type == "reg" and y_std_enable:
        y_te = (y_te - y_mu) / y_sigma

    loader = create_loader(emb_te, desc_te, y_te, bs=batch_size, shuffle=False)

    model, _tt = build_model_from_bundle(bundle, desc_te, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    preds = forward_predictions(
        model,
        loader,
        device,
        task_type,
        T if task_type == "cls" else None,
        mc_passes,
        y_std_enable,
        y_mu,
        y_sigma,
    )

    row_idx = np.array([data_dict[k]["_row_idx"] for k in sorted(data_dict.keys())], dtype=np.int64)
    full = np.full(n_total, np.nan, dtype=np.float64)
    full[row_idx] = preds

    labels = None
    if task_type == "cls" and thr is not None:
        labels = np.full(n_total, np.nan, dtype=np.float64)
        labels[row_idx] = (preds > thr).astype(np.float64)

    slug = task_info.get("task_name") or Path(bundle_path).parent.name
    return full, labels, str(slug), task_type, thr


def discover_checkpoint_jobs(checkpoints_root: str) -> List[Tuple[str, str, str]]:
    """
    每个子目录下匹配 stem 相同的 *.bundle.pt 与 *.pth。
    返回 list of (task_folder_name, bundle_path, ckpt_path)
    """
    root = Path(checkpoints_root)
    jobs = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for bundle in sorted(sub.glob("*.bundle.pt")):
            stem = bundle.name[: -len(".bundle.pt")]
            ckpt = sub / f"{stem}.pth"
            if ckpt.is_file():
                jobs.append((sub.name, str(bundle.resolve()), str(ckpt.resolve())))
    return jobs


def resolve_output_csv_path(
    out_dir: str,
    out_csv: str | None,
    mode: str,
    bundle_path: str | None,
) -> str:
    """
    统一落到「预测」目录（或可改的 out_dir）：
    - 未传 --out_csv：all_checkpoints -> 预测/admet_all_endpoints_preds.csv（**全任务一表**）
    - 未传 --out_csv：single -> 预测/{子目录名}_preds.csv（仅单任务调试用）
    - 只传文件名：放在 out_dir 下；带路径则按指定路径写
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not out_csv:
        if mode == "single":
            if not bundle_path:
                raise ValueError("single 模式需要 --bundle 以自动生成输出文件名")
            task_folder = Path(bundle_path).resolve().parent.name
            return str(Path(out_dir) / f"{task_folder}_preds.csv")
        return str(Path(out_dir) / "admet_all_endpoints_preds.csv")

    p = Path(out_csv)
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    if p.parent != Path("."):
        full = (Path.cwd() / p).resolve()
        full.parent.mkdir(parents=True, exist_ok=True)
        return str(full)
    return str(Path(out_dir) / p.name)


def build_parser():
    p = argparse.ArgumentParser("ArcMol 无标签预测（DeepFP 分块 pkl）")
    p.add_argument(
        "--mode",
        choices=["single", "all_checkpoints"],
        default="all_checkpoints",
        help="默认 all_checkpoints：checkpoints 下全部任务 → **一个**宽表 CSV；"
        "single：只跑一个 bundle（调试用，会单独一个 CSV）",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pkl_dir", type=str, default=None, help="DeepFP 生成的多个 *.pkl 所在目录（合并）"
    )
    src.add_argument(
        "--pkl_file",
        type=str,
        default=None,
        help="单个已合并的 *.pkl（顶层 dict，键为样本序号）",
    )
    p.add_argument("--bundle", type=str, default=None, help="[single] *.bundle.pt")
    p.add_argument("--ckpt", type=str, default=None, help="[single] *.pth，默认同 bundle 内 ckpt_path")
    p.add_argument("--checkpoints_root", type=str, default="checkpoints", help="[all] 检查点根目录")
    p.add_argument(
        "--out_dir",
        type=str,
        default="预测",
        help="预测 CSV 默认输出目录（默认在当前工作目录下创建「预测」文件夹）",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出 CSV 路径；可省略则自动写入 --out_dir；若只写文件名则放在 --out_dir 下",
    )
    p.add_argument(
        "--extra_attrs",
        type=str,
        default="SMILES",
        help="写入 CSV 的附加列（逗号分隔），如 SMILES",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--exclude_dir_prefix",
        action="append",
        default=[],
        help="跳过 checkpoints 子目录名以此前缀开头的任务（可多次指定）",
    )
    p.add_argument(
        "--no_skip_benchmark",
        action="store_true",
        help="不过滤 benchmark_* 目录（默认 all 模式会跳过）",
    )
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Out] 默认预测目录: {os.path.abspath(args.out_dir)}")

    if args.pkl_file:
        merged = load_merged_fp_pkl_file(args.pkl_file)
        print(f"[PKL] merged rows={len(merged)} from file {args.pkl_file}")
    else:
        merged = load_merged_fp_pkls(args.pkl_dir)
        print(f"[PKL] merged rows={len(merged)} from dir {args.pkl_dir}")
    n_total = len(merged)

    extra_attrs = [x.strip() for x in args.extra_attrs.split(",") if x.strip()]
    meta = extract_extra_attributes(merged, extra_attrs)

    if args.mode == "single":
        if not args.bundle:
            raise SystemExit("[single] 需要 --bundle")
        bundle = torch.load(args.bundle, map_location="cpu")
        ckpt = args.ckpt or bundle.get("ckpt_path")
        if not ckpt:
            raise SystemExit("请提供 --ckpt 或确保 bundle 含 ckpt_path")

        out_csv = resolve_output_csv_path(args.out_dir, args.out_csv, "single", args.bundle)

        preds, labels, slug, task_type, thr = run_one_bundle(
            args.bundle, ckpt, merged, n_total, device, extra_attrs, args.batch_size
        )

        df = pd.DataFrame()
        for a in extra_attrs:
            df[a] = meta.get(a, [None] * n_total)
        if task_type == "cls":
            df["pred_prob"] = preds
            if labels is not None:
                df["pred_label"] = labels
            if thr is not None:
                print(f"[{slug}] cls threshold={thr:.4f}")
        else:
            df["pred"] = preds
        df.to_csv(out_csv, index=False)
        print(f"[Saved] {os.path.abspath(out_csv)}  rows={len(df)}  task={slug}")
        return

    # ---------- all_checkpoints ----------
    ck_root = Path(args.checkpoints_root)
    if not ck_root.is_dir():
        raise SystemExit(f"checkpoints 目录不存在: {ck_root.resolve()}")

    out_csv = resolve_output_csv_path(args.out_dir, args.out_csv, "all_checkpoints", None)
    if args.exclude_dir_prefix:
        excl = tuple(args.exclude_dir_prefix)
    elif args.no_skip_benchmark:
        excl = ()
    else:
        excl = ("benchmark_",)

    run_all_checkpoints_wide(
        merged,
        str(ck_root),
        out_csv,
        extra_attrs=extra_attrs,
        batch_size=args.batch_size,
        device=device,
        exclude_dir_prefixes=excl,
    )


if __name__ == "__main__":
    main()
