# -*- coding: utf-8 -*-
"""
main_arcmol_mcc_r2.py  — training with selectable best-metric saving + inference bundle export.

What's new:
- Select which validation metric to save the "best" model on.
  * Classification: --selection_metric_cls {auc|f1|mcc}  (maximize)
  * Regression:     --selection_metric_reg {rmse|min, mae|min, r2|max}
- When saving best (classification), we recalibrate on VALID:
  temperature (T) by AUC, and threshold by F1 (or MCC if you selected MCC).
- Export a single bundle (*.bundle.pt) that contains everything for test-time:
  RDKit Top-K indices + scaler, RDKit attribute order, embedding order, model hparams,
  task info, label standardization params, MC-dropout passes, calibration, and ckpt path.
"""

import os, json, math, argparse, random, pickle
from typing import Tuple, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, matthews_corrcoef, r2_score
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from attention_pooling_fusion import AttentionPoolingFusion  # provided by your project


# -------------------------- Utils --------------------------
def set_seed(seed=24):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False


def _safe_flatten_1d(x):
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.dim() > 1: t = t.view(-1)
        return t.cpu()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.reshape(-1)).cpu()
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.float32).view(-1).cpu()
    else:
        return torch.tensor([float(x)], dtype=torch.float32).cpu()


def _bad(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor): x = _safe_flatten_1d(x)
    if x.numel()==0: return True
    if torch.isnan(x).any() or torch.isinf(x).any(): return True
    if torch.all(x==0): return True
    return False


# -------------------------- Data --------------------------
def filter_data_for_training(data_dict, embed_types, target='target'):
    if isinstance(embed_types, str): embed_types=[embed_types]
    out={}
    for k, rec in data_dict.items():
        if target not in rec or rec[target] is None:
            continue
        tv = rec[target]
        try:
            if isinstance(tv, float) and np.isnan(tv): continue
        except Exception:
            pass
        ok=True
        for et in embed_types:
            arr = rec.get(et, None)
            if arr is None: ok=False; break
            t = _safe_flatten_1d(arr)
            if _bad(t): ok=False; break
        if ok: out[k]=rec
    return out


def extract_rdkit_and_target(data_dict, attributes, target='target',
                             default_value=0.0, max_value=1e6, min_value=-1e6,
                             task_type='cls'):
    items=list(data_dict.values())
    X=[]
    for item in items:
        rd=item['rdkit_descriptors']
        feat=[]
        for a in attributes:
            v = rd.get(a, default_value)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): v=default_value
            feat.append(v)
        X.append(feat)
    X=np.nan_to_num(np.asarray(X,dtype=np.float64), nan=default_value, posinf=max_value, neginf=min_value)
    X=np.clip(X, min_value, max_value)
    if task_type == 'cls':
        y=np.array([int(item[target]) for item in items], dtype=np.int64)
    else:
        y=np.array([float(item[target]) for item in items], dtype=np.float32)
    return X, y


def select_top_k_features(X, y, k=210, task_type='cls'):
    if task_type == 'cls':
        scores = mutual_info_classif(X, y, random_state=0)
    else:
        scores = mutual_info_regression(X, y, random_state=0)
    idx = np.argsort(scores)[-k:]
    return X[:, idx], idx


def extract_selected_embedding(data_dict, embed_types, target='target', task_type='cls'):
    if isinstance(embed_types,str): embed_types=[embed_types]
    embs={et:[] for et in embed_types}; ys=[]
    for _, item in data_dict.items():
        if target not in item or item[target] is None: continue
        tv=item[target]
        try:
            if isinstance(tv,float) and np.isnan(tv): continue
        except Exception: pass
        cache={}; ok=True
        for et in embed_types:
            e=item.get(et,None)
            if e is None: ok=False; break
            t=_safe_flatten_1d(e).float()
            if _bad(t): ok=False; break
            cache[et]=t
        if ok:
            for et in embed_types: embs[et].append(cache[et])
            ys.append(int(tv) if task_type=='cls' else float(tv))
    if not ys: raise ValueError("No valid records.")
    for et in embed_types: embs[et]=torch.stack(embs[et])
    if task_type=='cls':
        ys=torch.tensor(ys,dtype=torch.long)
    else:
        ys=torch.tensor(ys,dtype=torch.float32)
    return embs, ys


class PairDataset(Dataset):
    def __init__(self, emb_dict, desc, y):
        self.emb=emb_dict; self.desc=desc; self.y=y
        n=len(y)
        for k in self.emb: assert len(self.emb[k])==n
        assert len(self.desc)==n
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return ({k:self.emb[k][i] for k in self.emb}, self.desc[i]), self.y[i]


def create_loader(emb, desc, y, bs=64, shuffle=True):
    return DataLoader(PairDataset(emb,desc,y), batch_size=bs, shuffle=shuffle)


# -------------------------- Modules --------------------------
class TaskAwareDescriptorPooling(nn.Module):
    def __init__(self, in_dim, h=128, out_dim=64, drop=0.1):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_dim,h), nn.ReLU(), nn.LayerNorm(h), nn.Dropout(drop))
        self.q = nn.Parameter(torch.randn(1,h)*0.02)
        self.val = nn.Linear(h,h)
        self.out = nn.Sequential(nn.Linear(h,out_dim), nn.ReLU(), nn.Dropout(drop))
    def forward(self, x):
        h=self.feat(x)
        q=self.q.expand(h.size(0),-1)
        a=torch.sigmoid((h*q).sum(-1,keepdim=True)/(h.size(-1)**0.5))
        v=self.val(h)
        pooled=a*v
        return self.out(pooled), a


class TaskContextProvider(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.ctx=nn.Parameter(torch.randn(1,d)*0.02)
    def forward(self,B,device): return self.ctx.to(device).expand(B,-1)


class ProxyMolHead(nn.Module):
    """Classification student head"""
    def __init__(self, in_dim, num_classes=2, margin=0.2, scale=32.0, drop=0.2):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(in_dim,in_dim), nn.PReLU(), nn.Dropout(drop))
        self.proxies=nn.Parameter(torch.randn(num_classes,in_dim)*0.1)
        self.margin=margin; self.scale=scale
    def forward(self,x,y=None,margin_scale=1.0):
        z=self.proj(x)
        z=nn.functional.normalize(z,dim=-1)
        p=nn.functional.normalize(self.proxies,dim=-1)
        logits=z@p.t()
        if y is not None:
            oh=torch.zeros_like(logits); oh.scatter_(1,y.view(-1,1),1.0)
            logits = logits - oh*(self.margin*margin_scale)
        return logits*self.scale, z


class MLPHead(nn.Module):
    """Classification teacher head"""
    def __init__(self,in_dim,hidden=128,num_classes=2,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden,num_classes))
    def forward(self,x): return self.net(x)


class RegHead(nn.Module):
    """Regression head"""
    def __init__(self,in_dim,hidden=128,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )
    def forward(self,x): return self.net(x)  # [B,1]


class ArcProxyRegularizer(nn.Module):
    """Optional Arc-style representation regularizer for regression."""
    def __init__(self, in_dim, nbins=32, drop=0.1, margin=0.10, scale=16.0,
                 y_min=-3.0, y_max=3.0, soft_sigma=0.0):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, in_dim), nn.PReLU(), nn.Dropout(drop))
        self.proxies = nn.Parameter(torch.randn(nbins, in_dim) * 0.1)
        self.margin = margin
        self.scale = scale
        self.nbins = nbins
        self.soft_sigma = soft_sigma
        bins = torch.linspace(y_min, y_max, nbins).view(nbins, 1)
        self.register_buffer("bin_values", bins)  # [K,1]
        self.adaptive_sigma = (soft_sigma<=0)

    @torch.no_grad()
    def update_bins(self, y_min, y_max):
        vals = torch.linspace(float(y_min), float(y_max), self.nbins, device=self.bin_values.device).view(self.nbins, 1)
        self.bin_values.copy_(vals)
        print(f"[ArcReg] bins -> [{float(y_min):.4f}, {float(y_max):.4f}], K={self.nbins}, sigma={'auto' if self.adaptive_sigma else self.soft_sigma}")

    def _soft_targets(self, y):
        if self.adaptive_sigma:
            bw = (self.bin_values[1] - self.bin_values[0]).abs().clamp_min(1e-6)
            sigma = float((bw / math.sqrt(2.0)).item())
        else:
            sigma = self.soft_sigma
        y = y.view(-1, 1)
        diff2 = (y - self.bin_values.T) ** 2  # [B,K]
        logits = - diff2 / (2.0 * (sigma ** 2) + 1e-8)
        return torch.softmax(logits, dim=1)

    def forward(self, x, y, margin_scale=1.0, class_balance=None):
        z = self.proj(x)
        z = nn.functional.normalize(z, dim=-1)
        p = nn.functional.normalize(self.proxies, dim=-1)
        logits = z @ p.t()  # [B,K]

        with torch.no_grad():
            nearest_idx = torch.argmin((y.view(-1, 1) - self.bin_values.T).abs(), dim=1)
        oh = torch.zeros_like(logits)
        oh.scatter_(1, nearest_idx.view(-1, 1), 1.0)
        logits = logits - oh * (self.margin * margin_scale)

        logits_scaled = logits * self.scale

        soft_true = self._soft_targets(y)  # [B,K]
        if class_balance is not None:
            w = class_balance.view(1, -1)
            soft_true = soft_true * w
            soft_true = soft_true / (soft_true.sum(dim=1, keepdim=True) + 1e-9)

        ce = -(soft_true * torch.log_softmax(logits_scaled, dim=1)).sum(dim=1).mean()
        return ce, z


class ArcMolModel(nn.Module):
    def __init__(self, fusion: AttentionPoolingFusion, desc: TaskAwareDescriptorPooling,
                 in_dim, task_type='cls', num_classes=2, task_ctx_dim=16, use_task_ctx=True,
                 margin=0.2, scale=32.0, head_hidden=128, head_dropout=0.3, proxy_dropout=0.2,
                 arc_reg_use=False, arc_reg_nbins=32, arc_reg_margin=0.10, arc_reg_scale=16.0, arc_reg_soft_sigma=0.0):
        super().__init__()
        self.fusion=fusion
        self.desc=desc
        self.task_type = task_type
        self.ctx = TaskContextProvider(task_ctx_dim) if use_task_ctx else None

        if task_type == 'cls':
            self.student=ProxyMolHead(in_dim,num_classes=num_classes,margin=margin,scale=scale,drop=proxy_dropout)
            self.teacher=MLPHead(in_dim,hidden=head_hidden,num_classes=num_classes,drop=head_dropout)
            self.arc_reg = None  # off for classification
        else:
            self.student=RegHead(in_dim,hidden=head_hidden,drop=proxy_dropout)
            self.teacher=RegHead(in_dim,hidden=head_hidden,drop=head_dropout)
            self.arc_reg = ArcProxyRegularizer(in_dim, nbins=arc_reg_nbins, drop=0.1,
                                               margin=arc_reg_margin, scale=arc_reg_scale,
                                               soft_sigma=arc_reg_soft_sigma) if arc_reg_use else None

    def encode(self, emb_dict, desc):
        B=desc.size(0); device=desc.device
        task_context = self.ctx(B,device) if self.ctx is not None else None
        fused, sparse_reg = self.fusion(emb_dict, task_context=task_context)
        drep, _ = self.desc(desc)
        x = torch.cat([fused, drep], dim=1)
        return x, sparse_reg

    def forward_logits(self, batch, y=None, margin_scale=1.0):
        (emb_dict,d) = batch
        x, sparse_reg = self.encode(emb_dict, d)

        if self.task_type == 'cls':
            ls, z = self.student(x, y=y, margin_scale=margin_scale)
            lt = self.teacher(x)
            return ls, lt, z, sparse_reg
        else:
            zs = self.student(x)
            zt = self.teacher(x)
            return zs, zt, None, sparse_reg


# -------------------------- Loss & Eval --------------------------
def ce_loss(logits,y,ls=0.02):
    return nn.functional.cross_entropy(logits,y,label_smoothing=ls)


def kd_loss_cls(ls, lt, T=2.0):
    ps = nn.functional.log_softmax(ls/T,dim=1)
    pt = nn.functional.softmax(lt/T,dim=1)
    return nn.functional.kl_div(ps,pt,reduction='batchmean')*(T*T)


def kd_loss_reg(zs, zt):
    return nn.functional.mse_loss(zs, zt.detach())


def cosine_warmup(epoch, warm):
    if warm<=0: return 1.0
    t=max(0,min(epoch/float(warm),1.0))
    return 0.5*(1-math.cos(math.pi*t))


@torch.no_grad()
def eval_epoch(model, loader, device, task_type='cls', temperature=None, mc_passes=8,
               y_std_enable=False, y_mu=0.0, y_sigma=1.0):
    model.eval()
    ys, preds = [], []
    logits_cache = []

    for (emb_dict,d), y in loader:
        emb_dict={k:v.to(device) for k,v in emb_dict.items()}
        d=d.to(device); y=y.to(device)

        out_sum = 0
        for _ in range(mc_passes):
            if task_type == 'cls':
                ls, lt, _, _ = model.forward_logits((emb_dict,d), y=None, margin_scale=1.0)
                logits = lt
                if temperature is not None: logits = logits/temperature
                out_sum += logits
            else:
                zs, zt, _, _ = model.forward_logits((emb_dict,d), y=None, margin_scale=1.0)
                out_sum += zt

        out = out_sum/mc_passes

        if task_type == 'cls':
            prob = torch.softmax(out, dim=1)[:,1]
            ys.append(y.detach().cpu().numpy())
            preds.append(prob.detach().cpu().numpy())
            logits_cache.append(out.detach().cpu().numpy())
        else:
            ys.append(y.detach().cpu().numpy())
            preds.append(out.detach().cpu().numpy().reshape(-1))

    y=np.concatenate(ys); p=np.concatenate(preds)

    if task_type == 'cls':
        auc=roc_auc_score(y,p)
        f1=f1_score(y,(p>0.5).astype(int))
        return dict(AUC=float(auc),F1=float(f1)), y, p, (np.concatenate(logits_cache,axis=0) if logits_cache else None)
    else:
        if y_std_enable:
            y_raw = y * y_sigma + y_mu
            p_raw = p * y_sigma + y_mu
        else:
            y_raw = y; p_raw = p
        rmse=float(np.sqrt(mean_squared_error(y_raw, p_raw)))
        mae =float(mean_absolute_error(y_raw, p_raw))
        r2 = float(r2_score(y_raw, p_raw))
        return dict(RMSE=rmse, MAE=mae, R2=r2), y_raw, p_raw, None


def temp_scaling_on_logits(y_val, logits_val, grid=np.linspace(0.5,20,200)):
    best_T, best_auc = 1.0, -1
    for T in grid:
        prob = torch.softmax(torch.from_numpy(logits_val)/T, dim=1)[:,1].numpy()
        auc = roc_auc_score(y_val, prob)
        if auc>best_auc: best_auc, best_T = auc, T
    return best_T


def choose_prob_thr(y, prob):
    cand=np.linspace(0.05,0.95,19)
    best_t, best=-1,-1
    for t in cand:
        s=f1_score(y,(prob>t).astype(int))
        if s>best: best=s; best_t=t
    return best_t


def choose_prob_thr_mcc(y, prob):
    """Find threshold that maximizes MCC on validation set; return (thr, best_mcc)."""
    cand=np.linspace(0.05,0.95,19)
    best_t, best=0.5,-1.0
    for t in cand:
        pred=(prob>t).astype(int)
        try:
            mcc = matthews_corrcoef(y, pred)
        except Exception:
            mcc = -1.0
        if mcc>best:
            best=mcc; best_t=t
    return best_t, best


def save_calib(path,T,thr):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w') as f: json.dump({'temperature':float(T),'prob_threshold':float(thr)},f)


def load_calib(path):
    with open(path,'r') as f: d=json.load(f)
    return float(d.get('temperature',1.0)), float(d.get('prob_threshold',0.5))


# -------------------------- Parser --------------------------
def build_arg_parser():
    p=argparse.ArgumentParser()
    # 数据
    p.add_argument('--data_dir', type=str, default='/home/data/zou/CMD-/CMD-ADMET/dataset/moleculeACE/CHEMBL3979_EC50')
    p.add_argument('--task_name', type=str, default='CHEMBL3979_EC50')
    p.add_argument('--task_type', type=str, choices=['cls','reg'], default='reg')
    p.add_argument('--valid_ratio', type=float, default=0.2,
                        help='当没有 valid.pkl 时，从 train.pkl 随机划分到验证集的比例（0~1），默认 0.2')

    p.add_argument('--fusion_embed_types',type=str,nargs='+',
        default=['RDKFingerprint','MACCSkeys','EStateFingerprint',
                 'MolT5','BioT5','AttrMask','GPT-GNN','GraphCL','MolCLR','GraphMVP',
                 'GROVER','UniMolV1','UniMolV2_84M','UniMolV2_1.1B'])
    p.add_argument('--target_name', type=str, default='target_y')

    # ---------------- 训练超参 ----------------
    p.add_argument('--k', type=int, default=210)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--freeze_epochs', type=int, default=10)
    p.add_argument('--margin_warmup_epochs', type=int, default=55)
    p.add_argument('--seed', type=int, default=42)

    # ---------------- MoE / 稀疏性优化 ----------------
    # 优化：从 5 降为 3，提高推理效率，捕捉最关键模态即可
    p.add_argument('--moe_topk', type=int, default=5)
    # 优化：从 0.3 降为 0.05，避免过度稀疏导致欠拟合
    p.add_argument('--moe_sparse_lambda', type=float, default=0.05)

    # ---------------- 学习率/正则 ----------------
    p.add_argument('--lr_fusion', type=float, default=2e-4)
    p.add_argument('--lr_head', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--max_grad_norm', type=float, default=1.0)

    # ---------------- 蒸馏与损失 ----------------
    # 优化：从 0.8 降为 0.5，让模型更关注 Ground Truth，减少对 Teacher 的过度依赖
    p.add_argument('--kd_max_lambda', type=float, default=0.5)
    p.add_argument('--label_smooth', type=float, default=0.1)

    # ---------------- 模态丢弃 (Modality Dropout) ----------------
    # 优化：从 0.45 大幅降为 0.20，由“近半丢失”改为“适度扰动”，显著提升训练稳定性
    p.add_argument('--moddrop_p', type=float, default=0.20)

    # ---------------- MC Dropout ----------------
    # 优化：从 12 降为 5，验证集评估速度提升一倍以上，足够用于训练期监控
    p.add_argument('--mc_passes', type=int, default=5)

    # ---------------- 任务上下文 ----------------
    p.add_argument('--use_task_ctx', action='store_true', default=True)
    p.add_argument('--task_ctx_dim', type=int, default=16)

    # ---------------- 分类/回归头 (Head) ----------------
    p.add_argument('--margin', type=float, default=0.2)
    p.add_argument('--scale', type=float, default=32.0)
    p.add_argument('--head_hidden', type=int, default=128)
    p.add_argument('--head_dropout', type=float, default=0.3)
    p.add_argument('--proxy_dropout', type=float, default=0.2)
    # 优化：降低正交约束，给予分类特征空间更多自由度
    p.add_argument('--cls_proxy_ortho_lambda', type=float, default=0.1)

    # ---------------- 融合模块 (Fusion) ----------------
    # [保持原样] 你要求的固定参数
    p.add_argument('--fusion_hidden_dim', type=int, default=64)
    p.add_argument('--fusion_out_dim', type=int, default=128)
    p.add_argument('--fusion_dropout', type=float, default=0.5)

    p.add_argument('--comp_mode', type=str, default='cka_rbf')
    # 优化：从 1e-3 提至 0.01，增强 RBF 核的区分度
    p.add_argument('--cka_gamma', type=float, default=0.01)
    p.add_argument('--task_gate', type=str, default='scalar')
    p.add_argument('--comp_scale', type=float, default=1.0)

    # ---------------- 标签标准化（仅回归） ----------------
    p.add_argument('--y_std_enable', action='store_true', default=True)
    p.add_argument('--y_std_eps', type=float, default=1e-6)

    # ---------------- Arc 表征正则（回归专用优化） ----------------
    p.add_argument('--arc_reg_use', action='store_true', default=True)
    p.add_argument('--arc_reg_lambda', type=float, default=0.10)
    p.add_argument('--arc_reg_nbins', type=int, default=32)
    p.add_argument('--arc_reg_margin', type=float, default=0.10)
    # 优化：Scale 提升至 32.0，增强梯度信号
    p.add_argument('--arc_reg_scale', type=float, default=32.0)
    # 优化：Sigma 降至 0.5，减少模糊，提高定位精度
    p.add_argument('--arc_reg_soft_sigma', type=float, default=0.5)
    p.add_argument('--arc_reg_hist_balance', action='store_true', default=True)

    # ---------------- 最佳模型选择策略 ----------------
    p.add_argument('--selection_metric_cls', type=str, default='auc',
                   choices=['auc', 'f1', 'mcc'],
                   help='classification: metric to MAXIMIZE when saving best')
    p.add_argument('--selection_metric_reg', type=str, default='rmse',
                   choices=['rmse', 'mae', 'r2'],
                   help='regression: rmse/mae MINIMIZE, r2 MAXIMIZE')

    # ---------------- 校准策略 ----------------
    p.add_argument('--use_valid_mcc_thr', action='store_true', default=False,
                   help='Use VALID-set MCC to choose prob threshold (with temp scaling). Default: fixed T=1.0, thr=0.5')

    p.add_argument('--save_dir', type=str, default='./runs_arc_reg')
    return p


# -------------------------- Training & Eval --------------------------
def _save_inference_bundle(bundle_path, *, args, attributes, topk_idx, scaler,
                           y_mu, y_sigma, task_type, target_name, fusion_embed_types,
                           model_hparams, mc_passes, calibration=None, ckpt_path=None, seed=24):
    """Save a single .bundle.pt with everything for test-only inference."""
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    bundle = {
        'version': 1,
        'seed': int(seed),
        'task': {'task_type': task_type, 'target_name': target_name},
        'fusion_embed_types': list(fusion_embed_types),
        'rdkit': {
            'attribute_names': list(attributes),
            'topk_idx': np.array(topk_idx, dtype=np.int64),
            'scaler': scaler,
        },
        'label_std': {
            'enable': bool(args.y_std_enable if task_type=='reg' else False),
            'mu': float(y_mu),
            'sigma': float(y_sigma),
        },
        'model_hparams': model_hparams,
        'mc_passes': int(mc_passes),
        'calibration': calibration,
        'ckpt_path': ckpt_path,
    }
    torch.save(bundle, bundle_path)
    lite = {k: v for k, v in bundle.items() if k != 'rdkit'}
    lite['rdkit'] = {'attribute_names': list(attributes), 'topk_idx_len': int(len(topk_idx))}
    with open(bundle_path + '.lite.json', 'w') as f:
        json.dump(lite, f, indent=2)
    print(f"[Bundle] saved -> {bundle_path} (+ .lite.json)")


def train_and_eval(args):
    """
    训练与评估入口：
    - 若 data_dir 下存在 <task>_valid.pkl 且过滤后非空，则使用之；
    - 否则从 <task>_train.pkl 经过过滤后的样本中按 8:2 随机切分出 valid。
    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 路径助手 ----------
    def _p(split: str) -> str:
        return os.path.join(args.data_dir, f"{args.task_name}_{split}.pkl")

    # ---------- 加载原始数据 ----------
    if not os.path.exists(_p('train')):
        raise FileNotFoundError(f"[Data] missing file: {_p('train')}")
    with open(_p('train'), 'rb') as f:
        train_raw = pickle.load(f)

    valid_raw = None
    if os.path.exists(_p('valid')):
        with open(_p('valid'), 'rb') as f:
            valid_raw = pickle.load(f)

    if not os.path.exists(_p('test')):
        raise FileNotFoundError(f"[Data] missing file: {_p('test')}")
    with open(_p('test'), 'rb') as f:
        test_raw = pickle.load(f)

    # ---------- 先过滤 train / test，valid 视情况处理 ----------
    full_train = filter_data_for_training(train_raw, args.fusion_embed_types, args.target_name)
    test_data  = filter_data_for_training(test_raw,  args.fusion_embed_types, args.target_name)

    if valid_raw is not None:
        valid_data = filter_data_for_training(valid_raw, args.fusion_embed_types, args.target_name)
    else:
        valid_data = {}

    # ---------- 若没有 valid 或过滤后 valid 为空：从 full_train 中按比例切分 ----------
    if (valid_raw is None) or (len(valid_data) == 0):
        keys = list(full_train.keys())
        if len(keys) < 5:
            raise ValueError(f"[Data] too few training samples ({len(keys)}) to split.")
        ratio = float(getattr(args, 'valid_ratio', 0.2))
        ratio = min(max(ratio, 0.0), 0.9)  # 简单防呆：避免全被分走
        rng = random.Random(args.seed)
        rng.shuffle(keys)
        n_valid = max(1, int(round(ratio * len(keys))))
        valid_keys = set(keys[:n_valid])
        train_keys = [k for k in keys if k not in valid_keys]
        train_data = {k: full_train[k] for k in train_keys}
        valid_data = {k: full_train[k] for k in valid_keys}
        print(f"[Split] No usable VALID set -> random {int((1 - ratio) * 100)}:{int(ratio * 100)} from train.pkl | "
              f"train={len(train_data)} | valid={len(valid_data)}")
    else:
        train_data = full_train
        print(f"[Use VALID] train={len(train_data)} | valid={len(valid_data)} | test={len(test_data)}")

    print(f"[After Filter] train={len(train_data)} | valid={len(valid_data)} | test={len(test_data)}")

    # ---------- RDKit 特征与 Top-K 选择 ----------
    attributes = list(train_data[list(train_data.keys())[0]]['rdkit_descriptors'].keys())

    Xtr, ytr = extract_rdkit_and_target(train_data, attributes, args.target_name, task_type=args.task_type)
    Xva, yva = extract_rdkit_and_target(valid_data, attributes, args.target_name, task_type=args.task_type)
    Xte, yte = extract_rdkit_and_target(test_data,  attributes, args.target_name, task_type=args.task_type)

    Xtr_sel, topk = select_top_k_features(Xtr, ytr, k=args.k, task_type=args.task_type)
    Xva_sel = Xva[:, topk]; Xte_sel = Xte[:, topk]

    scaler = StandardScaler()
    Xtr_sel = scaler.fit_transform(Xtr_sel)
    Xva_sel = scaler.transform(Xva_sel)
    Xte_sel = scaler.transform(Xte_sel)

    Xtr_sel = np.clip(Xtr_sel, -1e6, 1e6)
    Xva_sel = np.clip(Xva_sel, -1e6, 1e6)
    Xte_sel = np.clip(Xte_sel, -1e6, 1e6)

    desc_tr = torch.tensor(Xtr_sel, dtype=torch.float32)
    desc_va = torch.tensor(Xva_sel, dtype=torch.float32)
    desc_te = torch.tensor(Xte_sel, dtype=torch.float32)

    emb_tr, y_tr = extract_selected_embedding(train_data, args.fusion_embed_types, args.target_name, task_type=args.task_type)
    emb_va, y_va = extract_selected_embedding(valid_data, args.fusion_embed_types, args.target_name, task_type=args.task_type)
    emb_te, y_te = extract_selected_embedding(test_data,  args.fusion_embed_types, args.target_name, task_type=args.task_type)

    # ---------- （回归）标签标准化 ----------
    y_mu = 0.0; y_sigma = 1.0
    if args.task_type == 'reg' and args.y_std_enable:
        y_mu = float(torch.mean(y_tr.float()).item())
        y_sigma = float(torch.std(y_tr.float()).item())
        y_sigma = y_sigma if y_sigma > args.y_std_eps else 1.0
        y_tr = (y_tr - y_mu) / y_sigma
        y_va = (y_va - y_mu) / y_sigma
        y_te = (y_te - y_mu) / y_sigma

    # ---------- DataLoader ----------
    train_loader = create_loader(emb_tr, desc_tr, y_tr, bs=args.batch_size, shuffle=True)
    valid_loader = create_loader(emb_va, desc_va, y_va, bs=args.batch_size, shuffle=False)
    test_loader  = create_loader(emb_te, desc_te, y_te, bs=args.batch_size, shuffle=False)

    # ---------- 模型 ----------
    desc_module = TaskAwareDescriptorPooling(in_dim=desc_tr.shape[1], h=128, out_dim=64, drop=0.1).to(device)
    fusion_module = AttentionPoolingFusion(
        used_embedding_types=args.fusion_embed_types,
        l_output_dim=args.fusion_out_dim,
        hidden_dim=args.fusion_hidden_dim,
        dropout_prob=args.fusion_dropout,
        comp_mode=args.comp_mode, cka_gamma=args.cka_gamma,
        task_gate=args.task_gate, task_ctx_dim=args.task_ctx_dim,
        comp_scale=args.comp_scale,
        top_k=args.moe_topk, sparse_lambda=args.moe_sparse_lambda
    ).to(device)

    in_dim = args.fusion_out_dim + 64
    num_classes = 2 if args.task_type == 'cls' else 1
    model = ArcMolModel(
        fusion_module, desc_module, in_dim=in_dim,
        task_type=args.task_type, num_classes=num_classes,
        task_ctx_dim=args.task_ctx_dim, use_task_ctx=args.use_task_ctx,
        margin=args.margin, scale=args.scale,
        head_hidden=args.head_hidden, head_dropout=args.head_dropout,
        proxy_dropout=args.proxy_dropout,
        arc_reg_use=(args.arc_reg_use if args.task_type == 'reg' else False),
        arc_reg_nbins=args.arc_reg_nbins,
        arc_reg_margin=args.arc_reg_margin,
        arc_reg_scale=args.arc_reg_scale,
        arc_reg_soft_sigma=args.arc_reg_soft_sigma
    ).to(device)

    # ---------- 训练设定（与原版一致） ----------
    def freeze_fusion_desc(flag=True):
        for p in model.fusion.parameters(): p.requires_grad = not flag
        for p in model.desc.parameters():   p.requires_grad = not flag
    freeze_fusion_desc(True)

    params = [
        {'params': list(model.student.parameters()) + list(model.teacher.parameters()) +
                   (list(model.arc_reg.parameters()) if (hasattr(model, 'arc_reg') and model.arc_reg is not None) else []),
         'lr': args.lr_head, 'weight_decay': args.weight_decay},
        {'params': list(model.fusion.parameters()) + list(model.desc.parameters()),
         'lr': args.lr_fusion, 'weight_decay': args.weight_decay}
    ]
    opt = optim.Adam(params)

    # ---------- 选择最佳指标/保存 ----------
    if args.task_type == 'cls':
        maximize = True  # auc/f1/mcc 均为最大化
    else:
        maximize = (args.selection_metric_reg.lower() == 'r2')
    best_val_metric = -1e18 if maximize else 1e18
    best_epoch = -1; no_improve = 0
    tag = '+'.join(args.fusion_embed_types)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt  = os.path.join(args.save_dir, f'best_arcmol_arc_reg_{tag}_seed{args.seed}.pth')
    calib = os.path.join(args.save_dir, f'calib_{tag}_seed{args.seed}.json')
    bundle = os.path.join(args.save_dir, f'best_arcmol_arc_reg_{tag}_seed{args.seed}.bundle.pt')

    # ---------- 训练循环（保持与原版一致） ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch == args.freeze_epochs + 1:
            freeze_fusion_desc(False)

        run_loss = 0.0
        for (emb_dict, d), y in train_loader:
            emb_dict = {k: v.to(device) for k, v in emb_dict.items()}
            d = d.to(device); y = y.to(device)

            # 模态丢弃
            if args.moddrop_p > 0 and model.training:
                for k in list(emb_dict.keys()):
                    if random.random() < args.moddrop_p:
                        emb_dict[k] = torch.zeros_like(emb_dict[k])

            opt.zero_grad()
            mscale = cosine_warmup(epoch, args.margin_warmup_epochs)

            if args.task_type == 'cls':
                ls, lt, _, sparse_reg = model.forward_logits((emb_dict, d), y=y, margin_scale=mscale)
                loss_t = ce_loss(lt, y, ls=args.label_smooth)
                loss_s = ce_loss(ls, y, ls=args.label_smooth)
                lkd = kd_loss_cls(ls, lt.detach(), T=2.0)
                kd_lambda = args.kd_max_lambda * cosine_warmup(epoch, args.margin_warmup_epochs)
                loss = 0.5 * loss_t + 0.5 * loss_s + kd_lambda * lkd + sparse_reg
                if args.cls_proxy_ortho_lambda > 0.0:
                    P = nn.functional.normalize(model.student.proxies, dim=-1)
                    G = P @ P.t()
                    I = torch.eye(G.size(0), device=G.device)
                    loss_ortho = (G - I).pow(2).mean()
                    loss = loss + args.cls_proxy_ortho_lambda * loss_ortho
            else:
                zs, zt, _, sparse_reg = model.forward_logits((emb_dict, d), y=None, margin_scale=1.0)
                yf = y.view_as(zs).float()
                loss_t = nn.functional.mse_loss(zt, yf)
                loss_s = nn.functional.mse_loss(zs, yf)
                lkd = kd_loss_reg(zs, zt)
                kd_lambda = args.kd_max_lambda * cosine_warmup(epoch, args.margin_warmup_epochs)
                loss = 0.5 * loss_t + 0.5 * loss_s + kd_lambda * lkd + sparse_reg

                if model.arc_reg is not None and args.arc_reg_lambda > 0:
                    x, _ = model.encode(emb_dict, d)
                    arc_ce, _ = model.arc_reg(x, y.view(-1, 1).float(), margin_scale=mscale, class_balance=None)
                    loss = loss + args.arc_reg_lambda * arc_ce

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            run_loss += float(loss.item())

        # ---------- VALID ----------
        metrics, y_val, p_val, logits_val = eval_epoch(
            model, valid_loader, device,
            task_type=args.task_type, temperature=None, mc_passes=args.mc_passes,
            y_std_enable=(args.task_type == 'reg' and args.y_std_enable),
            y_mu=y_mu, y_sigma=y_sigma
        )

        if args.task_type == 'cls':
            auc = metrics['AUC']
            thr_f1 = choose_prob_thr(y_val, p_val)
            f1_best = f1_score(y_val, (p_val > thr_f1).astype(int))

            if args.use_valid_mcc_thr:
                thr_mcc, mcc_best = choose_prob_thr_mcc(y_val, p_val)
                mcc_view = mcc_best
                thr_mcc_view = thr_mcc
                mcc_tag = f"ValMCC* {mcc_view:.4f} (thr={thr_mcc_view:.2f})"
            else:
                thr_mcc_view = 0.50
                mcc_view = matthews_corrcoef(y_val, (p_val > thr_mcc_view).astype(int))
                mcc_best = mcc_view
                mcc_tag = f"ValMCC@{thr_mcc_view:.2f} {mcc_view:.4f}"

            sel = args.selection_metric_cls.lower()
            if   sel == 'auc': cur_metric = auc
            elif sel == 'f1':  cur_metric = f1_best
            elif sel == 'mcc': cur_metric = mcc_best
            else: raise ValueError(f'Unknown selection_metric_cls: {sel}')
            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"TrainLoss {run_loss/len(train_loader):.4f} | "
                  f"ValAUC {auc:.4f} | ValF1* {f1_best:.4f} (thr={thr_f1:.2f}) | "
                  f"{mcc_tag} | "
                  f"[Select={sel}, Cur={cur_metric:.4f}, Best={best_val_metric:.4f}]")

            better = (cur_metric > best_val_metric) if maximize else (cur_metric < best_val_metric)
            if better:
                best_val_metric = cur_metric; best_epoch = epoch; no_improve = 0
                torch.save(model.state_dict(), ckpt)

                # 校准：开关由 args.use_valid_mcc_thr 控制
                if args.use_valid_mcc_thr and (logits_val is not None):
                    T = temp_scaling_on_logits(y_val, logits_val)
                    prob_T = torch.softmax(torch.from_numpy(logits_val)/T, dim=1)[:, 1].numpy()
                    thr, _ = choose_prob_thr_mcc(y_val, prob_T)
                else:
                    T, thr = 1.0, 0.5
                save_calib(calib, T, thr)

                model_hparams = dict(
                    fusion_hidden_dim=args.fusion_hidden_dim, fusion_out_dim=args.fusion_out_dim,
                    fusion_dropout=args.fusion_dropout, comp_mode=args.comp_mode, cka_gamma=args.cka_gamma,
                    task_gate=args.task_gate, task_ctx_dim=args.task_ctx_dim, use_task_ctx=bool(args.use_task_ctx),
                    comp_scale=args.comp_scale, moe_topk=args.moe_topk, moe_sparse_lambda=args.moe_sparse_lambda,
                    head_hidden=args.head_hidden, head_dropout=args.head_dropout, proxy_dropout=args.proxy_dropout,
                    margin=args.margin, scale=args.scale,
                    arc_reg_use=False, arc_reg_nbins=args.arc_reg_nbins, arc_reg_margin=args.arc_reg_margin,
                    arc_reg_scale=args.arc_reg_scale, arc_reg_soft_sigma=args.arc_reg_soft_sigma
                )
                _save_inference_bundle(
                    bundle_path=bundle, args=args, attributes=attributes, topk_idx=topk, scaler=scaler,
                    y_mu=y_mu, y_sigma=y_sigma, task_type=args.task_type, target_name=args.target_name,
                    fusion_embed_types=args.fusion_embed_types, model_hparams=model_hparams,
                    mc_passes=args.mc_passes, calibration={'temperature': float(T), 'threshold': float(thr)},
                    ckpt_path=ckpt, seed=args.seed
                )
                print(f"  -> Saved best model & calib & bundle. [use_valid_mcc_thr={args.use_valid_mcc_thr}, T={T:.3f}, thr={thr:.3f}]")
            else:
                no_improve += 1

        else:
            rmse, mae, r2 = metrics['RMSE'], metrics['MAE'], metrics['R2']
            sel = args.selection_metric_reg.lower()
            if   sel == 'rmse': cur_metric = rmse
            elif sel == 'mae':  cur_metric = mae
            elif sel == 'r2':   cur_metric = r2
            else: raise ValueError(f'Unknown selection_metric_reg: {sel}')
            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"TrainLoss {run_loss/len(train_loader):.4f} | "
                  f"ValRMSE {rmse:.4f} | ValMAE {mae:.4f} | ValR2 {r2:.4f} | "
                  f"[Select={sel}, Cur={cur_metric:.4f}, Best={best_val_metric:.4f}]")

            better = (cur_metric > best_val_metric) if maximize else (cur_metric < best_val_metric)
            if better:
                best_val_metric = cur_metric; best_epoch = epoch; no_improve = 0
                torch.save(model.state_dict(), ckpt)

                model_hparams = dict(
                    fusion_hidden_dim=args.fusion_hidden_dim, fusion_out_dim=args.fusion_out_dim,
                    fusion_dropout=args.fusion_dropout, comp_mode=args.comp_mode, cka_gamma=args.cka_gamma,
                    task_gate=args.task_gate, task_ctx_dim=args.task_ctx_dim, use_task_ctx=bool(args.use_task_ctx),
                    comp_scale=args.comp_scale, moe_topk=args.moe_topk, moe_sparse_lambda=args.moe_sparse_lambda,
                    head_hidden=args.head_hidden, head_dropout=args.head_dropout, proxy_dropout=args.proxy_dropout,
                    margin=args.margin, scale=args.scale,
                    arc_reg_use=bool(args.arc_reg_use), arc_reg_nbins=args.arc_reg_nbins, arc_reg_margin=args.arc_reg_margin,
                    arc_reg_scale=args.arc_reg_scale, arc_reg_soft_sigma=args.arc_reg_soft_sigma
                )
                _save_inference_bundle(
                    bundle_path=bundle, args=args, attributes=attributes, topk_idx=topk, scaler=scaler,
                    y_mu=y_mu, y_sigma=y_sigma, task_type=args.task_type, target_name=args.target_name,
                    fusion_embed_types=args.fusion_embed_types, model_hparams=model_hparams,
                    mc_passes=args.mc_passes, calibration=None,
                    ckpt_path=ckpt, seed=args.seed
                )
                print("  -> Saved best model & bundle.")
            else:
                no_improve += 1

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    # ---------- 统一评估（保持原版） ----------
    @torch.no_grad()
    def load_and_eval(name, loader):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()

        if args.task_type == 'cls':
            T, thr = (1.0, 0.5)
            if os.path.exists(calib):
                with open(calib, 'r') as f: d = json.load(f)
                T = float(d.get('temperature', 1.0)); thr = float(d.get('prob_threshold', 0.5))

            metrics, y, p, _ = eval_epoch(model, loader, dev, task_type='cls', temperature=T, mc_passes=args.mc_passes)
            cov = (p > thr).mean()
            acc = ((((p > thr).astype(int) == y) & (p > thr)).sum()) / max(int((p > thr).sum()), 1)
            mcc = matthews_corrcoef(y, (p > thr).astype(int))
            f1  = f1_score(y, (p > thr).astype(int))

            print(f"\n=== {name} ===")
            print(f"{name:5s} AUC: {metrics['AUC']:.3f} | F1: {f1:.3f} | MCC: {mcc:.3f} | "
                  f"Coverage={cov:.3f}, Acc@accepted={acc:.3f} (prob_thr={thr:.2f})")
            return metrics['AUC']
        else:
            metrics, y_raw, p_raw, _ = eval_epoch(
                model, loader, dev, task_type='reg', temperature=None, mc_passes=args.mc_passes,
                y_std_enable=(args.task_type == 'reg' and args.y_std_enable), y_mu=y_mu, y_sigma=y_sigma
            )
            print(f"\n=== {name} ===")
            print(f"{name:5s} RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")
            return metrics['RMSE']

    test_metric  = load_and_eval("TEST",  test_loader)
    valid_metric = load_and_eval("VALID", valid_loader)
    _            = load_and_eval("TRAIN", train_loader)

    return dict(best_val=best_val_metric, best_test=test_metric, best_epoch=best_epoch,
                best_model_path=ckpt, bundle_path=bundle)


# -------------------------- main --------------------------
def main():
    parser=build_arg_parser()
    args=parser.parse_args()
    _ = train_and_eval(args)

if __name__=="__main__":
    main()
