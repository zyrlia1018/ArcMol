# -*- coding: utf-8 -*-
"""
test_only_arcmol.py  — bundle-based test-only inference.
You only provide: TEST split + bundle [+ optionally ckpt].
Everything else (TopK/Scaler/attribute order/embedding order/model hparams/calibration) comes from the bundle.
"""

import os
import argparse
import pickle
import numpy as np
import torch

from main_arcmol_mcc_r2 import (
    set_seed,
    filter_data_for_training,
    extract_rdkit_and_target,
    extract_selected_embedding,
    create_loader,
    eval_epoch,
    TaskAwareDescriptorPooling,
    ArcMolModel,
)
from attention_pooling_fusion import AttentionPoolingFusion
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, mean_squared_error, mean_absolute_error, \
    r2_score


def build_parser():
    p = argparse.ArgumentParser("ArcMol TEST-only (bundle version)")
    p.add_argument('--data_dir', type=str, required=True, help='directory containing {task_name}_test.pkl')
    p.add_argument('--task_name', type=str, required=True)
    p.add_argument('--bundle', type=str, required=True, help='*.bundle.pt exported during training')
    p.add_argument('--ckpt', type=str, default=None, help='optional; override ckpt_path inside bundle')
    p.add_argument('--save_preds', type=str, default=None)
    p.add_argument('--extra_attrs', type=str, default='SMILES',
                   help='comma-separated list of extra attributes to save, e.g., SMILES,cliff_mol')
    return p


def _load_split_pkl(data_dir, task_name, split):
    path = os.path.join(data_dir, f"{task_name}_{split}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_extra_attributes(data_dict, attrs):
    """
    Return dict[attr_name] = [values in same order as filtered data].
    Handles cases where items in data_dict might not be dicts (will append None),
    and looks up attributes in several common locations (top-level, 'rdkit_descriptors',
    case-insensitive keys like 'SMILES'/'smiles').
    """
    items = list(data_dict.values())
    out = {a: [] for a in attrs}
    for item in items:
        # default None for each attribute
        for a in attrs:
            val = None
            try:
                # If item is a dict-like mapping
                if isinstance(item, dict):
                    # direct hit
                    if a in item:
                        val = item.get(a)
                    else:
                        # case-insensitive try (e.g., 'SMILES' vs 'smiles')
                        if isinstance(a, str):
                            lowkeys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
                            if a.lower() in lowkeys:
                                val = item.get(lowkeys[a.lower()])

                        # maybe it's inside rdkit_descriptors
                        if val is None and 'rdkit_descriptors' in item and isinstance(item.get('rdkit_descriptors'), dict):
                            rd = item.get('rdkit_descriptors')
                            if a in rd:
                                val = rd.get(a)
                            else:
                                rd_map = {k.lower(): k for k in rd.keys() if isinstance(k, str)}
                                if isinstance(a, str) and a.lower() in rd_map:
                                    val = rd.get(rd_map[a.lower()])

                        # sometimes SMILES stored as 'smiles' or 'SMILES'
                        if val is None and isinstance(a, str) and a.lower() == 'smiles':
                            if 'smiles' in item:
                                val = item.get('smiles')
                            elif 'SMILES' in item:
                                val = item.get('SMILES')
                else:
                    # item not a dict (e.g., an int/float). In that case, we can't extract named attrs.
                    val = None
            except Exception:
                val = None
            out[a].append(val)
    return out



def main():
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 解析额外属性参数
    extra_attrs = [attr.strip() for attr in args.extra_attrs.split(',') if attr.strip()]
    print(f"[Extra Attributes] Will save: {extra_attrs}")

    # 1) load bundle
    bundle = torch.load(args.bundle, map_location='cpu')
    assert int(bundle.get('version', 1)) >= 1, "Unsupported bundle version"
    seed = int(bundle.get('seed', 24))
    set_seed(seed)
    task_info = bundle['task']
    task_type = task_info['task_type']
    target_name = task_info['target_name']
    fusion_embed_types = bundle['fusion_embed_types']
    rdkit_meta = bundle['rdkit']
    attributes = rdkit_meta['attribute_names']
    topk = rdkit_meta['topk_idx']
    scaler = rdkit_meta['scaler']
    label_std = bundle.get('label_std', {'enable': False, 'mu': 0.0, 'sigma': 1.0})
    y_std_enable = bool(label_std.get('enable', False)) and (task_type == 'reg')
    y_mu = float(label_std.get('mu', 0.0))
    y_sigma = float(label_std.get('sigma', 1.0))
    model_hparams = bundle['model_hparams']
    mc_passes = int(bundle.get('mc_passes', 12))
    calib = bundle.get('calibration', None)
    ckpt = args.ckpt if args.ckpt is not None else bundle.get('ckpt_path', None)
    if ckpt is None:
        raise ValueError("Provide --ckpt or ensure bundle contains ckpt_path")

    print("[Bundle] task_type:", task_type, "| target:", target_name)
    print("[Bundle] embeds:", "+".join(fusion_embed_types))
    print("[Bundle] RDKit TopK =", len(topk))

    # 2) TEST data
    test_data = _load_split_pkl(args.data_dir, args.task_name, 'test')
    test_data = filter_data_for_training(test_data, fusion_embed_types, target=target_name)
    print(f"[After Filter] test={len(test_data)}")

    # 提取额外属性
    extra_attrs_data = extract_extra_attributes(test_data, extra_attrs)

    # 3) RDKit -> select TopK -> scale (using bundle's objects)
    Xte, yte = extract_rdkit_and_target(test_data, attributes, target=target_name, task_type=task_type)
    Xte_sel = Xte[:, topk]
    Xte_sel = scaler.transform(Xte_sel)
    Xte_sel = np.clip(Xte_sel, -1e6, 1e6)
    desc_te = torch.tensor(Xte_sel, dtype=torch.float32)

    # 4) Embeddings & labels
    emb_te, y_te = extract_selected_embedding(test_data, fusion_embed_types, target=target_name, task_type=task_type)

    if task_type == 'reg' and y_std_enable:
        y_te = (y_te - y_mu) / y_sigma

    # 5) loader
    test_loader = create_loader(emb_te, desc_te, y_te, bs=128, shuffle=False)

    # 6) model from bundle
    desc_module = TaskAwareDescriptorPooling(in_dim=desc_te.shape[1], h=128, out_dim=64, drop=0.1).to(device)
    fusion_module = AttentionPoolingFusion(
        used_embedding_types=fusion_embed_types,
        l_output_dim=model_hparams['fusion_out_dim'],
        hidden_dim=model_hparams['fusion_hidden_dim'],
        dropout_prob=model_hparams['fusion_dropout'],
        comp_mode=model_hparams['comp_mode'], cka_gamma=model_hparams['cka_gamma'],
        task_gate=model_hparams['task_gate'], task_ctx_dim=model_hparams['task_ctx_dim'],
        comp_scale=model_hparams['comp_scale'],
        top_k=model_hparams['moe_topk'], sparse_lambda=model_hparams['moe_sparse_lambda']
    ).to(device)

    in_dim = model_hparams['fusion_out_dim'] + 64
    num_classes = 2 if task_type == 'cls' else 1
    model = ArcMolModel(
        fusion_module, desc_module, in_dim=in_dim,
        task_type=task_type, num_classes=num_classes,
        task_ctx_dim=model_hparams['task_ctx_dim'], use_task_ctx=bool(model_hparams['use_task_ctx']),
        margin=model_hparams['margin'], scale=model_hparams['scale'],
        head_hidden=model_hparams['head_hidden'], head_dropout=model_hparams['head_dropout'],
        proxy_dropout=model_hparams['proxy_dropout'],
        arc_reg_use=bool(model_hparams['arc_reg_use']),
        arc_reg_nbins=model_hparams['arc_reg_nbins'],
        arc_reg_margin=model_hparams['arc_reg_margin'],
        arc_reg_scale=model_hparams['arc_reg_scale'],
        arc_reg_soft_sigma=model_hparams['arc_reg_soft_sigma'],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 7) eval
    if task_type == 'cls':
        T = float(calib.get('temperature', 1.0)) if calib else 1.0
        thr = float(calib.get('threshold', 0.5)) if calib else 0.5
        print(f"[Calib] T={T:.3f}, thr={thr:.3f}")
        metrics, y, p, _ = eval_epoch(
            model, test_loader, device,
            task_type='cls', temperature=T, mc_passes=mc_passes
        )
        yhat = (p > thr).astype(int)
        cov = (p > thr).mean()
        acc = (((yhat == y) & (p > thr))).sum() / max((p > thr).sum(), 1)
        mcc = matthews_corrcoef(y, yhat)
        f1 = f1_score(y, yhat)
        auc = roc_auc_score(y, p)
        print("\n=== TEST (CLS) ===")
        print(f"AUC: {auc:.3f} | F1: {f1:.3f} | MCC: {mcc:.3f} | Coverage={cov:.3f}, Acc@accepted={acc:.3f}")
        p_to_save = p
        y_to_save = y
    else:
        metrics, y_raw, p_raw, _ = eval_epoch(
            model, test_loader, device,
            task_type='reg', temperature=None, mc_passes=mc_passes,
            y_std_enable=y_std_enable, y_mu=y_mu, y_sigma=y_sigma
        )
        print("\n=== TEST (REG) ===")
        print(f"RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")
        p_to_save = p_raw
        y_to_save = y_raw

    # 8) save predictions with extra attributes
    if args.save_preds:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)
        if task_type == 'cls':
            df = pd.DataFrame({'y_true': y_to_save, 'prob': p_to_save})
        else:
            df = pd.DataFrame({'y_true': y_to_save, 'y_pred': p_to_save})

        # 添加额外属性到DataFrame
        for attr_name in extra_attrs:
            df[attr_name] = extra_attrs_data[attr_name]

        df.to_csv(args.save_preds, index=False)
        print(f"[Saved] {args.save_preds} with extra attributes: {extra_attrs}")


if __name__ == "__main__":
    main()