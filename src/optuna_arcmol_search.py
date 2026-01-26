
# -*- coding: utf-8 -*-
"""
Optuna search harness for ArcMol (compatible with main_arcmol_mcc_r2.py).
- Imports build_arg_parser/train_and_eval from main_arcmol_mcc_r2.py
- Each trial writes into <save_root>/trial_XXXX/
- After optimization, copies the best trial's artifacts (ckpt/bundle/calib)
  into <save_root>/best_<metric>/ for one-command testing.
"""

import argparse, json, os, shutil, sys, glob
from typing import Any, Dict

try:
    import optuna
except Exception:
    raise SystemExit("Please install Optuna first: pip install optuna")

# Import from the current training script
from main_arcmol_mcc_r2 import build_arg_parser, train_and_eval  # noqa


def _copy_best_artifacts(best_user_attrs: Dict[str, Any], save_root: str,
                         task_type: str, selection_metric_cls: str, selection_metric_reg: str):
    """Copy best ckpt/bundle/calib (+lite.json) into best_<metric>/ folder."""
    metric = selection_metric_cls if task_type == 'cls' else selection_metric_reg
    best_dir = os.path.join(save_root, f"best_{metric.lower()}")
    os.makedirs(best_dir, exist_ok=True)

    ckpt = best_user_attrs.get("best_model_path") or best_user_attrs.get("best_model")  # backward compat
    bundle = best_user_attrs.get("bundle_path")
    copied = []

    if bundle and os.path.isfile(bundle):
        dst = os.path.join(best_dir, os.path.basename(bundle))
        shutil.copy2(bundle, dst)
        copied.append(dst)
        # copy lite json if exists
        lite = bundle + ".lite.json"
        if os.path.isfile(lite):
            shutil.copy2(lite, os.path.join(best_dir, os.path.basename(lite)))
            copied.append(os.path.join(best_dir, os.path.basename(lite)))

    if ckpt and os.path.isfile(ckpt):
        dst = os.path.join(best_dir, os.path.basename(ckpt))
        shutil.copy2(ckpt, dst)
        copied.append(dst)
        # try to copy calib_* json in the same folder
        for jj in glob.glob(os.path.join(os.path.dirname(ckpt), "calib_*.json")):
            shutil.copy2(jj, os.path.join(best_dir, os.path.basename(jj)))
            copied.append(os.path.join(best_dir, os.path.basename(jj)))

    # write a small README with summary
    readme = os.path.join(best_dir, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Best artifacts copied by optuna_arcmol_search.py\n")
        f.write(f"Metric bucket: best_{metric.lower()}\n")
        f.write("Copied files:\n")
        for c in copied:
            f.write(f" - {os.path.basename(c)}\n")
        f.write("\nUse for testing, e.g.:\n")
        f.write("python test_only_arcmol.py --data_dir <DATA_DIR> --task_name <TASK> "
                f"--bundle {os.path.join(best_dir, os.path.basename(bundle) if bundle else '<bundle>.bundle.pt')} "
                "--save_preds out/test_preds.csv\n")
    print(f"[Best] Artifacts copied to: {best_dir}")
    return best_dir


def objective(trial: "optuna.trial.Trial", base_args: argparse.Namespace):
    # Copy args and override per trial
    args = argparse.Namespace(**vars(base_args))

    # ---- General search space ----
    args.lr_fusion = trial.suggest_float("lr_fusion", 5e-5, 5e-4, log=True)
    args.lr_head   = trial.suggest_float("lr_head",   1e-4, 3e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)
    args.label_smooth = trial.suggest_float("label_smooth", 0.00, 0.12)
    args.kd_max_lambda = trial.suggest_float("kd_max_lambda", 0.2, 1.0)
    args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.0, 3.0)

    args.batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
    args.moddrop_p  = trial.suggest_float("moddrop_p", 0.1, 0.6)
    args.mc_passes  = trial.suggest_categorical("mc_passes", [8, 12, 16])

    args.margin = trial.suggest_float("margin", 0.0, 0.40)
    args.scale  = trial.suggest_float("scale",  8.0, 40.0)
    args.head_hidden  = trial.suggest_categorical("head_hidden",  [128, 192, 256])
    args.head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
    args.proxy_dropout= trial.suggest_float("proxy_dropout", 0.0, 0.4)
    # Classification regularizer
    args.cls_proxy_ortho_lambda = trial.suggest_float("cls_proxy_ortho_lambda", 0.0, 0.01)

    args.fusion_hidden_dim = trial.suggest_categorical("fusion_hidden_dim", [64, 96, 128])
    args.fusion_out_dim    = trial.suggest_categorical("fusion_out_dim",    [128, 160, 192])
    args.fusion_dropout    = trial.suggest_float("fusion_dropout", 0.1, 0.6)
    args.cka_gamma         = trial.suggest_float("cka_gamma", 1e-4, 5e-3, log=True)
    args.comp_mode         = trial.suggest_categorical("comp_mode", ['cka_rbf'])
    args.task_gate         = trial.suggest_categorical("task_gate", ['scalar'])

    # MoE
    args.moe_topk          = trial.suggest_categorical("moe_topk", [3, 5, 7])
    args.moe_sparse_lambda = trial.suggest_float("moe_sparse_lambda", 0.0, 0.05)

    # Regression-only knobs (harmless for cls due to guards inside trainer)
    args.y_std_enable      = trial.suggest_categorical("y_std_enable", [True, False])
    args.arc_reg_use       = trial.suggest_categorical("arc_reg_use", [True, False])
    args.arc_reg_lambda    = trial.suggest_float("arc_reg_lambda", 0.02, 0.20)
    args.arc_reg_nbins     = trial.suggest_categorical("arc_reg_nbins", [16, 24, 32, 40])
    args.arc_reg_margin    = trial.suggest_float("arc_reg_margin", 0.00, 0.20)
    args.arc_reg_scale     = trial.suggest_float("arc_reg_scale", 8.0, 32.0)
    args.arc_reg_soft_sigma= trial.suggest_float("arc_reg_soft_sigma", 0.0, 1.0)
    args.arc_reg_hist_balance = trial.suggest_categorical("arc_reg_hist_balance", [True, False])

    # Per-trial save dir
    args.save_dir = os.path.join(base_args.save_root, f"trial_{trial.number:04d}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Run one training/validation
    result = train_and_eval(args)
    # result is a dict from trainer: {'best_val', 'best_test', 'best_epoch', 'best_model_path', 'bundle_path'}

    # Record attributes for analysis
    trial.set_user_attr("best_val",   float(result.get('best_val')))
    trial.set_user_attr("best_test",  float(result.get('best_test')))
    trial.set_user_attr("best_epoch", int(result.get('best_epoch')))
    trial.set_user_attr("best_model_path", result.get('best_model_path'))
    trial.set_user_attr("bundle_path",     result.get('bundle_path'))

    # Objective value: best_test (AUC for cls; RMSE for reg)
    return float(result.get('best_test'))


def main():
    parser = build_arg_parser()
    # add Optuna-specific flags
    parser.add_argument('--study_name', type=str, default='arcmol_opt')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage, e.g. sqlite:///arcmol.db; None=in-memory')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--save_root', type=str, default='arcmol_study_runs')
    # (the base parser already includes: data_dir/task_name/task_type/target_name/selection_metric_*)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    # Direction aligned with what trainer returns (AUC ↑ for cls, RMSE ↓ for reg)
    direction = "maximize" if args.task_type == "cls" else "minimize"

    if args.storage:
        study = optuna.create_study(study_name=args.study_name, storage=args.storage,
                                    load_if_exists=True, direction=direction)
    else:
        study = optuna.create_study(direction=direction)

    print("Starting Optuna study:", study.study_name)
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("Best Value:", best.value)
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("Attrs:")
    for k, v in best.user_attrs.items():
        print(f"  {k}: {v}")

    # Save summary for reproducibility
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "direction": direction,
        "task_type": args.task_type,
        "selection_metric_cls": getattr(args, 'selection_metric_cls', None),
        "selection_metric_reg": getattr(args, 'selection_metric_reg', None),
    }
    with open(os.path.join(args.save_root, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved best summary to", os.path.join(args.save_root, "best_summary.json"))

    # Copy best artifacts to <save_root>/best_<metric>/
    best_dir = _copy_best_artifacts(best.user_attrs, args.save_root,
                                    args.task_type, getattr(args, 'selection_metric_cls', 'auc'),
                                    getattr(args, 'selection_metric_reg', 'rmse'))
    print("Best artifacts dir:", best_dir)


if __name__ == "__main__":
    main()


