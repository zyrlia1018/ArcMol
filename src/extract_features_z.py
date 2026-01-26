# -*- coding: utf-8 -*-
"""
extract_features.py  — Modified to extract Z (Normalized Latent Features)
for unit sphere projection, instead of raw X (concatenated features).
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

# --- 注意：这些依赖项需要您的 main_arcmol_mcc_r2.py 和 attention_pooling_fusion.py
# --- 在运行时提供完整的类定义，否则会报 NameError。

# 从 main_arcmol_mcc_r2 导入所需的模块/函数/类
from main_arcmol_mcc_r2 import (
    set_seed,
    filter_data_for_training,
    extract_rdkit_and_target,
    extract_selected_embedding,
    create_loader,
    TaskAwareDescriptorPooling,
    ArcMolModel,
    # 假设 eval_epoch 在 main_arcmol_mcc_r2 中被正确定义或省略
)
from attention_pooling_fusion import AttentionPoolingFusion
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score  # 保持评估依赖


# [START: Modified eval_epoch or equivalent extraction function]
@torch.no_grad()
def extract_features_epoch_Z(model, loader, device, task_type='cls', mc_passes=1):
    """
    修改后的提取函数：获取归一化潜在特征 Z。
    """
    model.eval()
    Z_normalized_list = []
    Y_true_list = []

    # 确定要获取投影 (proj) 模块的头
    proj_module = None
    if task_type == 'cls' and hasattr(model, 'student') and hasattr(model.student, 'proj'):
        proj_module = model.student.proj
    elif task_type == 'reg' and hasattr(model, 'arc_reg') and model.arc_reg is not None and hasattr(model.arc_reg,
                                                                                                    'proj'):
        proj_module = model.arc_reg.proj

    if proj_module is None:
        print(f"[ERROR] Cannot find proj head for Z extraction (Task={task_type}).")
        return None, None

    for (emb_dict, d), y in loader:
        emb_dict = {k: v.to(device) for k, v in emb_dict.items()}
        d = d.to(device);
        y = y.to(device)

        # 1. 获取拼接特征 X (model.encode 的输出)
        x, _ = model.encode(emb_dict, d)

        # 2. 核心修改：应用投影层并归一化 (生成单位特征 Z)
        z_raw = proj_module(x)
        z_normalized = F.normalize(z_raw, dim=-1)

        Z_normalized_list.append(z_normalized.detach().cpu().numpy())
        Y_true_list.append(y.detach().cpu().numpy())

    Z_combined = np.concatenate(Z_normalized_list, axis=0)
    Y_combined = np.concatenate(Y_true_list)

    return Z_combined, Y_combined


# [END: Modified extraction function]


def build_parser():
    p = argparse.ArgumentParser("ArcMol Feature Extractor (Extract Z)")
    p.add_argument('--data_dir', type=str, required=True, help='directory containing {task_name}_{split}.pkl files')
    p.add_argument('--task_name', type=str, required=True)
    p.add_argument('--bundle', type=str, required=True, help='*.bundle.pt exported during training')
    p.add_argument('--ckpt', type=str, default=None, help='optional; override ckpt_path inside bundle')
    p.add_argument('--output_dir', type=str, default='./z_features_output_reg',
                   help='directory to save the extracted Z features (.pkl files)')
    p.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    p.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                   choices=['train', 'valid', 'test'],
                   help='Specify which data splits to process. Default: all three.')
    return p


def _load_split_pkl(data_dir, task_name, split):
    path = os.path.join(data_dir, f"{task_name}_{split}.pkl")
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)


def process_split(split_name, raw_data, bundle, model, device, args):
    if raw_data is None or not raw_data:
        print(f"[Warning] Skipping '{split_name}': Data file missing or empty.")
        return

    print(f"\n--- Processing Split: {split_name.upper()} ---")

    # 1) 解包 Bundle 参数
    task_type = bundle['task']['task_type']
    target_name = bundle['task']['target_name']
    fusion_embed_types = bundle['fusion_embed_types']
    attributes = bundle['rdkit']['attribute_names']
    topk = bundle['rdkit']['topk_idx']
    scaler = bundle['rdkit']['scaler']

    label_std = bundle.get('label_std', {'enable': False, 'mu': 0.0, 'sigma': 1.0})
    y_std_enable = bool(label_std.get('enable', False)) and (task_type == 'reg')
    y_mu = float(label_std.get('mu', 0.0))
    y_sigma = float(label_std.get('sigma', 1.0))

    # 2) 过滤数据
    filtered_data = filter_data_for_training(raw_data, fusion_embed_types, target=target_name)
    print(f"[{split_name.upper()}] Samples after filter: {len(filtered_data)}")
    if len(filtered_data) == 0:
        print(f"[Warning] Skipping '{split_name}': No valid samples remain after filtering.")
        return

    # 3) RDKit 特征准备
    X, y_raw_labels = extract_rdkit_and_target(filtered_data, attributes, target=target_name, task_type=task_type)
    X_sel = X[:, topk]
    X_sel = scaler.transform(X_sel)
    desc = torch.tensor(X_sel, dtype=torch.float32)
    emb, y_labels_for_loader = extract_selected_embedding(filtered_data, fusion_embed_types, target=target_name,
                                                          task_type=task_type)

    # 4) 应用标签标准化 (用于模型输入)
    if task_type == 'reg' and y_std_enable:
        y_labels_for_loader = (y_labels_for_loader - y_mu) / y_sigma

    # 5) DataLoader
    loader = create_loader(emb, desc, y_labels_for_loader, bs=args.batch_size, shuffle=False)

    # 6) 提取 Z 特征
    Z_features, Y_dummy = extract_features_epoch_Z(model, loader, device, task_type=task_type)

    # 7) 保存结果
    if Z_features is not None:
        # 保存时，使用原始标签 y_raw_labels
        output_path = os.path.join(args.output_dir, f"{args.task_name}_{split_name}_z_features.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({'features': Z_features, 'y_true': y_raw_labels}, f)

        print(f"[{split_name.upper()}] Saved Z features to: {output_path}")
        print(f"[{split_name.upper()}] Z Features shape: {Z_features.shape}")
    else:
        print(f"[{split_name.upper()}] Failed to extract Z features.")


def main():
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) 加载 Bundle
    bundle = torch.load(args.bundle, map_location='cpu')
    seed = int(bundle.get('seed', 24));
    set_seed(seed)
    task_type = bundle['task']['task_type']
    fusion_embed_types = bundle['fusion_embed_types']
    rdkit_meta = bundle['rdkit'];
    topk = rdkit_meta['topk_idx']
    model_hparams = bundle['model_hparams']
    ckpt = args.ckpt if args.ckpt is not None else bundle.get('ckpt_path', None)
    if ckpt is None: raise ValueError("Provide --ckpt or ensure bundle contains ckpt_path")

    # 2) 构建模型
    in_dim_desc = len(topk)
    # [WARNING] 依赖于外部 TaskAwareDescriptorPooling 和 AttentionPoolingFusion 的存在
    desc_module = TaskAwareDescriptorPooling(in_dim=in_dim_desc, h=128, out_dim=64, drop=0.1).to(device)
    fusion_module = AttentionPoolingFusion(
        used_embedding_types=fusion_embed_types, l_output_dim=model_hparams['fusion_out_dim'],
        hidden_dim=model_hparams['fusion_hidden_dim'], dropout_prob=model_hparams['fusion_dropout'],
        comp_mode=model_hparams['comp_mode'], cka_gamma=model_hparams['cka_gamma'],
        task_gate=model_hparams['task_gate'], task_ctx_dim=model_hparams['task_ctx_dim'],
        comp_scale=model_hparams['comp_scale'], top_k=model_hparams['moe_topk'],
        sparse_lambda=model_hparams['moe_sparse_lambda']
    ).to(device)

    in_dim = model_hparams['fusion_out_dim'] + 64
    num_classes = 2 if task_type == 'cls' else 1
    # [WARNING] 依赖于外部 ArcMolModel 及其所有子结构的完整定义
    model = ArcMolModel(
        fusion_module, desc_module, in_dim=in_dim, task_type=task_type, num_classes=num_classes,
        task_ctx_dim=model_hparams['task_ctx_dim'], use_task_ctx=bool(model_hparams['use_task_ctx']),
        margin=model_hparams['margin'], scale=model_hparams['scale'], head_hidden=model_hparams['head_hidden'],
        head_dropout=model_hparams['head_dropout'], proxy_dropout=model_hparams['proxy_dropout'],
        arc_reg_use=bool(model_hparams.get('arc_reg_use', False)),
        arc_reg_nbins=model_hparams.get('arc_reg_nbins', 32),
        arc_reg_margin=model_hparams.get('arc_reg_margin', 0.1),
        arc_reg_scale=model_hparams.get('arc_reg_scale', 16.0),
        arc_reg_soft_sigma=model_hparams.get('arc_reg_soft_sigma', 0.0),
    ).to(device)

    # 3) 加载权重
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 4) 逐个处理指定的数据集划分
    for split_name in args.splits:
        raw_data = _load_split_pkl(args.data_dir, args.task_name, split_name)
        process_split(split_name, raw_data, bundle, model, device, args)

    print("\n✅ All specified Z feature extractions complete.")


if __name__ == "__main__":
    main()
