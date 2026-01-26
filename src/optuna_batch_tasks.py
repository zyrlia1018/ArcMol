#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Optuna runner for ArcMol.

- Strictly read tasks CSV with comma delimiter.
- Required columns: task_name,data_dir,target_name
  Optional columns: task_type (cls/reg), dataset
- If task_type missing -> infer from pkl (test -> train -> valid).
- For each task:
    * build args from main_arcmol_mcc_r2.build_arg_parser()
    * run optuna_arcmol_search.objective(trial, args)
    * dump per-task best_summary.json
    * copy best artifacts into best_<metric>/ (AUC for cls, RMSE for reg)
- Finally write batch_best_summary.csv under save_root.
"""

import argparse
import csv
import json
import os
import sys
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import pickle

try:
    import optuna
except Exception as e:
    raise SystemExit("Please install optuna: pip install optuna") from e

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

# Use your existing modules
from main_arcmol_mcc_r2 import build_arg_parser  # 与单任务训练保持一致
from optuna_arcmol_search import objective       # 直接复用你的 objective()

# ------------------------------
# CSV (comma) reader
# ------------------------------
REQUIRED_COLS = ["task_name", "data_dir", "target_name"]

def read_tasks_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"tasks_csv not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=","))  # 严格逗号
    if not rows:
        raise ValueError(f"Empty tasks file: {csv_path}")
    # 规范化并校验
    norm: List[Dict[str, str]] = []
    for r in rows:
        r = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items() }
        for c in REQUIRED_COLS:
            if not r.get(c):
                raise ValueError(f"Row missing required column '{c}': {r}")
        norm.append(r)
    return norm

# ------------------------------
# Infer task_type from data_dir if missing (cls/reg)
# ------------------------------
def _load_pickle(p: str):
    with open(p, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            return pickle.load(f, encoding="latin1")

def _samples_from_obj(obj):
    if isinstance(obj, dict):  return [v for v in obj.values() if isinstance(v, dict)]
    if isinstance(obj, list):  return [v for v in obj if isinstance(v, dict)]
    return []

def _extract_targets_dict(sample: Dict) -> Optional[Dict]:
    t = sample.get("Targets")
    if isinstance(t, dict) and t:
        return t
    for k in ("target_label", "label", "y", "target"):
        if k in sample:
            return {k: sample[k]}
    return None

def _infer_target_key(samples: List[Dict]) -> Optional[str]:
    freq = {}
    for sm in samples:
        td = _extract_targets_dict(sm)
        if not td:
            continue
        for k in td.keys():
            freq[k] = freq.get(k, 0) + 1
    return max(freq, key=freq.get) if freq else None

def _collect_numeric_values(samples: List[Dict], target_key: str) -> List[float]:
    vals = []
    for sm in samples:
        td = _extract_targets_dict(sm)
        if not td or target_key not in td:
            continue
        v = td[target_key]
        if isinstance(v, (bool, int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            try:
                vals.append(float(v))
            except Exception:
                pass
        elif isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (bool, int, float)):
            vals.append(float(v[0]))
    return vals

def _is_cls(values: List[float]) -> bool:
    if not values:
        return False
    uniq = set()
    for v in values:
        if abs(v - 0.0) < 1e-8: uniq.add(0.0)
        elif abs(v - 1.0) < 1e-8: uniq.add(1.0)
        else: uniq.add(v)
        if len(uniq) > 2:
            return False
    return uniq.issubset({0.0, 1.0})

def _find_split_pkl(task_dir: str, task_name: str, split: str) -> Optional[str]:
    exact = os.path.join(task_dir, f"{task_name}_{split}.pkl")
    if os.path.isfile(exact):
        return exact
    for fn in os.listdir(task_dir):
        if fn.endswith(f"_{split}.pkl"):
            p = os.path.join(task_dir, fn)
            if os.path.isfile(p):
                return p
    return None

def infer_task_type_from_dir(data_dir: str, task_name: str) -> Optional[str]:
    last_tkey = None
    for split in ("test", "train", "valid"):
        pkl = _find_split_pkl(data_dir, task_name, split)
        if not pkl:
            continue
        try:
            obj = _load_pickle(pkl)
            samples = _samples_from_obj(obj)
            if not samples:
                continue
            tkey = _infer_target_key(samples) or "target_label"
            last_tkey = tkey
            vals = _collect_numeric_values(samples, tkey)
            if len(vals) > 0:
                return "cls" if _is_cls(vals) else "reg"
        except Exception:
            continue
    # 有 key 但没数值，保守回归
    return "reg" if last_tkey is not None else None

# ------------------------------
# Optuna helpers
# ------------------------------
def make_study(direction: str, storage_uri: Optional[str], study_name: str) -> optuna.Study:
    if storage_uri:
        os.makedirs(os.path.dirname(storage_uri.replace("sqlite:///", "")) or ".", exist_ok=True)
        return optuna.create_study(study_name=study_name, storage=storage_uri,
                                   load_if_exists=True, direction=direction)
    return optuna.create_study(direction=direction)

def copy_best_artifacts(user_attrs: Dict[str, Any], task_save_root: str,
                        task_type: str, sel_cls: str, sel_reg: str) -> str:
    """将 best 模型与 bundle 复制到 best_<metric>/，返回目标目录。"""
    metric = "auc" if task_type == "cls" else "rmse"
    best_dir = os.path.join(task_save_root, f"best_{metric}")
    os.makedirs(best_dir, exist_ok=True)

    for key in ["best_model_path", "bundle_path"]:
        p = user_attrs.get(key)
        if p and os.path.exists(p):
            try:
                shutil.copy2(p, os.path.join(best_dir, os.path.basename(p)))
            except Exception as e:
                print(f"[WARN] copy {key} failed: {p} ({e})")
    # 同步保存一份 summary
    with open(os.path.join(best_dir, "user_attrs.json"), "w", encoding="utf-8") as f:
        json.dump(user_attrs, f, ensure_ascii=False, indent=2)
    return best_dir

def run_one_task(task: Dict[str, Any], base_args: argparse.Namespace,
                 n_trials: int, save_root: str, storage_root: Optional[str],
                 seed: int) -> Dict[str, Any]:
    task_name   = task["task_name"]
    data_dir    = task["data_dir"]
    target_name = task["target_name"]
    dataset     = task.get("dataset", task_name)
    task_type   = (task.get("task_type") or "").lower()

    if task_type not in ("cls", "reg"):
        inferred = infer_task_type_from_dir(data_dir, task_name) or "cls"
        print(f"[INFO] task_type missing for {task_name}, inferred as '{inferred}'")
        task_type = inferred

    # 克隆基础参数并覆写
    args = argparse.Namespace(**vars(base_args))
    args.task_name   = task_name
    args.data_dir    = data_dir
    args.target_name = target_name
    args.task_type   = task_type

    # 每个任务独立保存目录
    task_save_root = os.path.join(save_root, task_name)
    os.makedirs(task_save_root, exist_ok=True)
    # 某些训练脚本使用 save_root 作为根；若你的训练用 save_dir，这里也可同时赋值
    args.save_root = task_save_root
    if not hasattr(args, "save_dir"):
        args.save_dir = task_save_root

    # 方向：分类最大化(AUC)；回归最小化(RMSE)
    direction = "maximize" if task_type == "cls" else "minimize"
    study_name = f"{task_name}_arcmol_opt"

    storage_uri = None
    if storage_root:
        os.makedirs(storage_root, exist_ok=True)
        storage_uri = f"sqlite:///{os.path.join(storage_root, study_name + '.db')}"

    # 随机种子（若 parser 支持）
    try:
        args.seed = seed
    except Exception:
        pass

    study = make_study(direction, storage_uri, study_name)
    study.optimize(lambda t: objective(t, args), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    ua = best.user_attrs  # objective() 内应写入 best_* 信息
    best_dir = copy_best_artifacts(
        ua,
        task_save_root,
        task_type,
        getattr(args, 'selection_metric_cls', 'auc'),
        getattr(args, 'selection_metric_reg', 'rmse'),
    )

    # 汇总
    summary = {
        "task_name": task_name,
        "task_type": task_type,
        "metric": "AUC" if task_type == "cls" else "RMSE",
        "best_value": best.value,
        "best_val": ua.get("best_val"),
        "best_test": ua.get("best_test"),
        "best_epoch": ua.get("best_epoch"),
        "best_model_path": ua.get("best_model_path"),
        "bundle_path": ua.get("bundle_path"),
        "best_params": best.params,
        "n_trials": len(study.trials),
        "study_name": study_name,
        "direction": direction,
        "dataset": dataset,
        "data_dir": data_dir,
        "target_name": target_name,
        "artifacts_dir": best_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(task_save_root, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def write_batch_summary(rows: List[Dict[str, Any]], out_csv: str) -> None:
    if not rows:
        return
    cols = [
        "task_name", "task_type", "metric", "best_test", "best_val", "best_value",
        "best_epoch", "best_model_path", "bundle_path",
        "n_trials", "study_name", "direction",
        "dataset", "data_dir", "target_name",
        "artifacts_dir", "timestamp", "best_params_json"
    ]
    for r in rows:
        r["best_params_json"] = json.dumps(r.get("best_params", {}), ensure_ascii=False)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_csv", type=str, required=True,
                    help="Comma-separated CSV with columns: task_name,data_dir,target_name[,task_type,dataset]")
    ap.add_argument("--save_root", type=str, default="arcmol_batch_runs",
                    help="Root dir for all task outputs")
    ap.add_argument("--storage_root", type=str, default=None,
                    help="If set, create per-task sqlite DBs under this directory")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    # 先解析批处理参数，其余透传到训练 parser
    args, unknown = ap.parse_known_args()

    tasks_rows = read_tasks_csv(args.tasks_csv)

    # 构建基础训练参数（与单任务一致），允许通过命令行覆盖
    train_parser = build_arg_parser()
    base_args = train_parser.parse_args(unknown)

    os.makedirs(args.save_root, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for i, row in enumerate(tasks_rows, 1):
        task_cfg = {
            "task_name": row["task_name"],
            "data_dir": row["data_dir"],
            "target_name": row["target_name"],
            "task_type": (row.get("task_type") or "").lower(),
            "dataset": row.get("dataset", row["task_name"]),
        }
        print("\n" + "=" * 80)
        print(f"[Task {i}/{len(tasks_rows)}] {task_cfg['task_name']} "
              f"(type={task_cfg.get('task_type') or 'auto'}, target={task_cfg['target_name']})")
        print("=" * 80)
        summary = run_one_task(
            task=task_cfg,
            base_args=base_args,
            n_trials=args.n_trials,
            save_root=args.save_root,
            storage_root=args.storage_root,
            seed=args.seed,
        )
        results.append(summary)

    out_csv = os.path.join(args.save_root, "batch_best_summary.csv")
    write_batch_summary(results, out_csv)
    print(f"\n[OK] Batch summary saved -> {out_csv}")

if __name__ == "__main__":
    main()
