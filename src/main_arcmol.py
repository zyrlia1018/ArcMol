
import os, json, math, argparse, random, pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from attention_pooling_fusion import AttentionPoolingFusion  # 原实现（已扩展 MoE）


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
        if target not in rec or rec[target] is None: continue
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

def select_top_k_features(X, y, k=210, task_type='cls', random_state=0):
    rs = 0 if random_state is None else int(random_state)
    if task_type == 'cls':
        scores = mutual_info_classif(X, y, random_state=rs)
    else:
        scores = mutual_info_regression(X, y, random_state=rs)
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
    """分类学生头（保持原逻辑，不改）"""
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
    """分类教师头（保持原逻辑，不改）"""
    def __init__(self,in_dim,hidden=128,num_classes=2,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden,num_classes))
    def forward(self,x): return self.net(x)

class RegHead(nn.Module):
    """回归头（原逻辑）"""
    def __init__(self,in_dim,hidden=128,drop=0.3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )
    def forward(self,x): return self.net(x)  # [B,1]


# ===== ArcFace-style 表征正则（只在回归使用；不影响分类路径） =====
class ArcProxyRegularizer(nn.Module):
    """
    ArcFace 风味的“值锚点”正则：仅作为辅助项，不做预测。
    - 沿原始 y 范围均匀放置 K 个锚点；
    - 对融合表示做 proj+L2 normalize，与锚点求余弦；
    - 对最邻近锚点施加 additive margin；
    - 用 y 生成高斯软标签做 CE；支持直方图均衡重权。
    """
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
    """
    TwoHeadModel -> ArcMol（分类路径保持不变；回归路径保持 teacher/student，但允许额外 Arc 正则）
    """
    def __init__(self, fusion: AttentionPoolingFusion, desc: TaskAwareDescriptorPooling,
                 in_dim, task_type='cls', num_classes=2, task_ctx_dim=16, use_task_ctx=True,
                 margin=0.2, scale=32.0, head_hidden=128, head_dropout=0.3, proxy_dropout=0.2,
                 # 新增：仅回归使用的 Arc 正则设置
                 arc_reg_use=False, arc_reg_nbins=32, arc_reg_margin=0.10, arc_reg_scale=16.0, arc_reg_soft_sigma=0.0):
        super().__init__()
        self.fusion=fusion
        self.desc=desc
        self.task_type = task_type
        self.ctx = TaskContextProvider(task_ctx_dim) if use_task_ctx else None

        if task_type == 'cls':
            # 分类：完全保持原逻辑（学生=ProxyMolHead，教师=MLPHead）
            self.student=ProxyMolHead(in_dim,num_classes=num_classes,margin=margin,scale=scale,drop=proxy_dropout)
            self.teacher=MLPHead(in_dim,hidden=head_hidden,num_classes=num_classes,drop=head_dropout)
            self.arc_reg = None  # 分类不启用
        else:
            # 回归：保持原 teacher/student（均为回归头），并可选 Arc 正则
            self.student=RegHead(in_dim,hidden=head_hidden,drop=proxy_dropout)
            self.teacher=RegHead(in_dim,hidden=head_hidden,drop=head_dropout)
            self.arc_reg = ArcProxyRegularizer(in_dim, nbins=arc_reg_nbins, drop=0.1,
                                               margin=arc_reg_margin, scale=arc_reg_scale,
                                               soft_sigma=arc_reg_soft_sigma) if arc_reg_use else None

    def encode(self, emb_dict, desc):
        B=desc.size(0); device=desc.device
        task_context = self.ctx(B,device) if self.ctx is not None else None
        fused, sparse_reg = self.fusion(emb_dict, task_context=task_context)  # (fused, reg)
        drep, _ = self.desc(desc)
        x = torch.cat([fused, drep], dim=1)
        return x, sparse_reg

    def forward_logits(self, batch, y=None, margin_scale=1.0):
        (emb_dict,d) = batch
        x, sparse_reg = self.encode(emb_dict, d)

        if self.task_type == 'cls':
            ls, z = self.student(x, y=y, margin_scale=margin_scale)  # [B,C]
            lt = self.teacher(x)                                     # [B,C]
            return ls, lt, z, sparse_reg
        else:
            zs = self.student(x)   # [B,1]
            zt = self.teacher(x)   # [B,1]
            return zs, zt, None, sparse_reg  # 回归分支不改变返回签名


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
    logits_cache = []  # for classification calibration (if needed)

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
                out_sum += zt  # regression: teacher as prediction

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
        # 如果启用标准化，则把 y 和 p 均反标准化回原始空间再评估
        if y_std_enable:
            y_raw = y * y_sigma + y_mu
            p_raw = p * y_sigma + y_mu
        else:
            y_raw = y; p_raw = p
        rmse=float(np.sqrt(mean_squared_error(y_raw, p_raw)))
        mae =float(mean_absolute_error(y_raw, p_raw))
        return dict(RMSE=rmse, MAE=mae), y_raw, p_raw, None

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
    p.add_argument('--data_dir', type=str, default='/home/data/zou/CMD-/CMD-ADMET/CMD-ADMET/freesolv/data')
    p.add_argument('--task_name', type=str, default='freesolv')
    p.add_argument('--task_type', type=str, choices=['cls','reg'], default='reg')

    p.add_argument('--fusion_embed_types',type=str,nargs='+',
        default=['RDKFingerprint','MACCSkeys','EStateFingerprint',
                 'MolT5','BioT5','AttrMask','GPT-GNN','GraphCL','MolCLR','GraphMVP',
                 'GROVER','UniMolV1','UniMolV2_84M','UniMolV2_1.1B'])
    p.add_argument('--target_name', type=str, default='target',
                   choices=['target', 'target_HIV_active',
                            'target_nr-ahr', 'target_nr-ar-lbd', 'target_nr-aromatase', 'target_nr-ar',
                            'target_nr-er-lbd', 'target_nr-er', 'target_nr-ppar-gamma', 'target_sr-are',
                            'target_sr-atad5', 'target_sr-hse', 'target_sr-mmp', 'target_sr-p53',
                            'target_ear and labyrinth disorders', 'target_blood and lymphatic system disorders',
                            'target_endocrine disorders', 'target_injury, poisoning and procedural complications',
                            'target_nervous system disorders', 'target_reproductive system and breast disorders',
                            'target_social circumstances',
                            'target_neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                            'target_cardiac disorders', 'target_respiratory, thoracic and mediastinal disorders',
                            'target_surgical and medical procedures', 'target_hepatobiliary disorders',
                            'target_gastrointestinal disorders', 'target_eye disorders',
                            'target_vascular disorders', 'target_congenital, familial and genetic disorders',
                            'target_investigations', 'target_psychiatric disorders',
                            'target_infections and infestations',
                            'target_musculoskeletal and connective tissue disorders',
                            'target_metabolism and nutrition disorders',
                            'target_general disorders and administration site conditions',
                            'target_pregnancy, puerperium and perinatal conditions',
                            'target_skin and subcutaneous tissue disorders',
                            'target_renal and urinary disorders', 'target_immune system disorders',
                            'target_product issues'
                            ],
                   )

    # 训练流程
    p.add_argument('--k',type=int,default=210)
    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--patience',type=int,default=30)
    p.add_argument('--freeze_epochs',type=int,default=10)
    p.add_argument('--margin_warmup_epochs',type=int,default=55)
    p.add_argument('--seed',type=int,default=24)

    # MoE / 稀疏
    p.add_argument('--moe_topk', type=int, default=5)
    p.add_argument('--moe_sparse_lambda', type=float, default=0.0)

    # 学习率/正则
    p.add_argument('--lr_fusion',type=float,default=2e-4)
    p.add_argument('--lr_head',type=float,default=1e-4)
    p.add_argument('--weight_decay',type=float,default=1e-3)
    p.add_argument('--max_grad_norm',type=float,default=0.0)

    # 蒸馏与损失（分类/回归主干保持不变）
    p.add_argument('--kd_max_lambda',type=float,default=0.8)
    p.add_argument('--label_smooth',type=float,default=0.08)

    # 模态丢弃
    p.add_argument('--moddrop_p',type=float,default=0.45)

    # MC dropout 次数
    p.add_argument('--mc_passes',type=int,default=12)

    # 任务上下文
    p.add_argument('--use_task_ctx',action='store_true',default=True)
    p.add_argument('--task_ctx_dim',type=int,default=16)

    # 分类学生头超参（保持原设置）
    p.add_argument('--cls_proxy_ortho_lambda', type=float, default=0.0,
                   help='分类：Proxy 原型正交正则权重（0 关闭；推荐 1e-3~5e-3）')
    p.add_argument('--margin',type=float,default=0.2)
    p.add_argument('--scale',type=float,default=32.0)
    p.add_argument('--head_hidden',type=int,default=128)
    p.add_argument('--head_dropout',type=float,default=0.3)
    p.add_argument('--proxy_dropout',type=float,default=0.2)

    # 融合模块
    p.add_argument('--fusion_hidden_dim',type=int,default=64)
    p.add_argument('--fusion_out_dim',type=int,default=128)
    p.add_argument('--fusion_dropout',type=float,default=0.5)
    p.add_argument('--comp_mode',type=str,default='cka_rbf')
    p.add_argument('--cka_gamma',type=float,default=1e-3)
    p.add_argument('--task_gate',type=str,default='scalar')
    p.add_argument('--comp_scale',type=float,default=1.0)

    # === 新增：仅回归时启用 Arc 表征正则 ===
    # 标签标准化（仅回归）：用训练集均值/方差 z-score
    p.add_argument('--y_std_enable', action='store_true', default=True,
                   help='仅在回归时启用标签标准化（z-score），评估与打印会自动反标准化到原始空间')
    p.add_argument('--y_std_eps', type=float, default=1e-6,
                   help='避免除零的小常数')

    p.add_argument('--arc_reg_use', action='store_true', default=True,
                   help='只在 task_type=reg 时生效；分类不使用')
    p.add_argument('--arc_reg_lambda', type=float, default=0.10,
                   help='Arc 正则损失的权重（相对 SmoothL1/MSE）')
    p.add_argument('--arc_reg_nbins', type=int, default=32,
                   help='值锚点个数，越小越稳')
    p.add_argument('--arc_reg_margin', type=float, default=0.10,
                   help='Arc 的 additive margin（建议小）')
    p.add_argument('--arc_reg_scale', type=float, default=16.0,
                   help='Arc 的 logits 放大系数（建议适中）')
    p.add_argument('--arc_reg_soft_sigma', type=float, default=0.0,
                   help='0 表示按 bin 宽自动设 σ（更稳）')
    p.add_argument('--arc_reg_hist_balance', action='store_true', default=True,
                   help='用直方图逆频率对软标签重标定')
    p.add_argument('--save_dir', type=str, default='./runs_arc_reg')
    return p


# -------------------------- 训练&评估 --------------------------
class ResultPack:
    def __init__(self, best_val, best_test, best_epoch, best_model_path):
        self.best_val = best_val
        self.best_test = best_test
        self.best_epoch = best_epoch
        self.best_model_path = best_model_path

def train_and_eval(args):
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    def _p(split: str) -> str:
        return os.path.join(args.data_dir, f"{args.task_name}_{split}.pkl")

    train_path = _p('train')
    valid_path = _p('valid')
    test_path  = _p('test')

    for pth in [train_path, valid_path, test_path]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"[Data] 找不到数据文件: {pth}")

    with open(train_path, 'rb') as f: train_data = pickle.load(f)
    with open(valid_path, 'rb') as f: valid_data = pickle.load(f)
    with open(test_path,  'rb') as f: test_data  = pickle.load(f)

    print(f"Train data size before filtering: {len(train_data)}")
    print(f"Valid data size before filtering: {len(valid_data)}")
    print(f"Test  data size before filtering: {len(test_data)}")

    train_data=filter_data_for_training(train_data,args.fusion_embed_types,args.target_name)
    valid_data=filter_data_for_training(valid_data,args.fusion_embed_types,args.target_name)
    test_data =filter_data_for_training(test_data, args.fusion_embed_types,args.target_name)

    print(f"Train data size after filtering: {len(train_data)}")
    print(f"Valid data size after filtering: {len(valid_data)}")
    print(f"Test  data size after filtering: {len(test_data)}")

    attributes=list(train_data[list(train_data.keys())[0]]['rdkit_descriptors'].keys())

    # RDKit 特征 + 目标
    Xtr,ytr = extract_rdkit_and_target(train_data,attributes,args.target_name, task_type=args.task_type)
    Xva,yva = extract_rdkit_and_target(valid_data,attributes,args.target_name, task_type=args.task_type)
    Xte,yte = extract_rdkit_and_target(test_data, attributes,args.target_name, task_type=args.task_type)

    Xtr_sel, topk = select_top_k_features(
        Xtr, ytr, k=args.k, task_type=args.task_type, random_state=args.seed
    )
    Xva_sel = Xva[:,topk]
    Xte_sel = Xte[:,topk]

    scaler=StandardScaler()
    Xtr_sel=scaler.fit_transform(Xtr_sel)
    Xva_sel=scaler.transform(Xva_sel)
    Xte_sel=scaler.transform(Xte_sel)

    Xtr_sel=np.clip(Xtr_sel,-1e6,1e6)
    Xva_sel=np.clip(Xva_sel,-1e6,1e6)
    Xte_sel=np.clip(Xte_sel,-1e6,1e6)

    desc_tr=torch.tensor(Xtr_sel,dtype=torch.float32)
    desc_va=torch.tensor(Xva_sel,dtype=torch.float32)
    desc_te=torch.tensor(Xte_sel,dtype=torch.float32)

    emb_tr, y_tr = extract_selected_embedding(train_data,args.fusion_embed_types,args.target_name, task_type=args.task_type)
    emb_va, y_va = extract_selected_embedding(valid_data,args.fusion_embed_types,args.target_name, task_type=args.task_type)
    emb_te, y_te = extract_selected_embedding(test_data, args.fusion_embed_types,args.target_name, task_type=args.task_type)

        # ===== 标签标准化（仅回归） =====
    y_mu = 0.0; y_sigma = 1.0
    if args.task_type == 'reg' and args.y_std_enable:
        y_mu = float(torch.mean(y_tr.float()).item())
        y_sigma = float(torch.std(y_tr.float()).item())
        y_sigma = y_sigma if y_sigma > args.y_std_eps else 1.0
        print(f"[LabelStd] enable z-score on labels: mu={y_mu:.6f}, sigma={y_sigma:.6f}")
        # standardize y for loaders
        y_tr = (y_tr - y_mu) / y_sigma
        y_va = (y_va - y_mu) / y_sigma
        y_te = (y_te - y_mu) / y_sigma

    # 重新构造 loaders 以使用（可能标准化后的）y
    train_loader=create_loader(emb_tr,desc_tr,y_tr,bs=args.batch_size,shuffle=True)
    valid_loader=create_loader(emb_va,desc_va,y_va,bs=args.batch_size,shuffle=False)
    test_loader =create_loader(emb_te,desc_te,y_te,bs=args.batch_size,shuffle=False)

    # ====== 仅回归：准备 Arc 正则的 y 范围与直方图权重 ======
    arc_class_balance=None; ymin=ymax=None
    if args.task_type == 'reg' and args.arc_reg_use:
        ymin, ymax = float(torch.min(y_tr.float()).item()), float(torch.max(y_tr.float()).item())
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.5
        ymin -= pad; ymax += pad
        # 这里的 y_tr 已是标准化后的（若启用），因此 Arc bins 与损失在同一坐标系

    # 模块
    desc_module=TaskAwareDescriptorPooling(in_dim=desc_tr.shape[1],h=128,out_dim=64,drop=0.1).to(device)
    fusion_module=AttentionPoolingFusion(
        used_embedding_types=args.fusion_embed_types,
        l_output_dim=args.fusion_out_dim,
        hidden_dim=args.fusion_hidden_dim,
        dropout_prob=args.fusion_dropout,
        comp_mode=args.comp_mode, cka_gamma=args.cka_gamma,
        task_gate=args.task_gate, task_ctx_dim=args.task_ctx_dim,
        comp_scale=args.comp_scale,
        top_k=args.moe_topk,
        sparse_lambda=args.moe_sparse_lambda
    ).to(device)

    in_dim = args.fusion_out_dim + 64
    num_classes = 2 if args.task_type=='cls' else 1
    model = ArcMolModel(fusion_module,desc_module,in_dim=in_dim,
                        task_type=args.task_type, num_classes=num_classes,
                        task_ctx_dim=args.task_ctx_dim,use_task_ctx=args.use_task_ctx,
                        margin=args.margin, scale=args.scale,
                        head_hidden=args.head_hidden, head_dropout=args.head_dropout,
                        proxy_dropout=args.proxy_dropout,
                        arc_reg_use=(args.arc_reg_use if args.task_type=='reg' else False),
                        arc_reg_nbins=args.arc_reg_nbins,
                        arc_reg_margin=args.arc_reg_margin,
                        arc_reg_scale=args.arc_reg_scale,
                        arc_reg_soft_sigma=args.arc_reg_soft_sigma
                        ).to(device)

    # 初始化 Arc 正则的 bins 和直方图均衡权重（只在回归）
    if args.task_type == 'reg' and args.arc_reg_use and (model.arc_reg is not None):
        model.arc_reg.update_bins(ymin, ymax)
        if args.arc_reg_hist_balance:
            with torch.no_grad():
                bins = model.arc_reg.bin_values.view(-1).cpu().numpy()
                edges = (bins[1:] + bins[:-1]) / 2.0
                edges = np.concatenate(([bins[0] - (edges[0]-bins[0])], edges, [bins[-1] + (bins[-1]-edges[-1])]))
                hist, _ = np.histogram(y_tr.cpu().numpy(), bins=edges)
                hist = hist + 1  # 平滑
                inv = 1.0 / hist.astype(np.float64)
                inv = inv / inv.mean()
                arc_class_balance = torch.tensor(inv, dtype=torch.float32, device=device)
                print("[ArcReg balance] counts:", hist.tolist())
                print("[ArcReg balance] weights(avg=1):", inv.tolist())

    # 冻结策略 & 优化器
    def freeze_fusion_desc(flag=True):
        for p in model.fusion.parameters(): p.requires_grad = not flag
        for p in model.desc.parameters():   p.requires_grad = not flag
    freeze_fusion_desc(True)

    params=[
        {'params': list(model.student.parameters())+list(model.teacher.parameters())+
                   (list(model.arc_reg.parameters()) if (hasattr(model,'arc_reg') and model.arc_reg is not None) else []),
         'lr': args.lr_head, 'weight_decay': args.weight_decay},
        {'params': list(model.fusion.parameters())+list(model.desc.parameters()),
         'lr': args.lr_fusion, 'weight_decay': args.weight_decay}
    ]
    opt=optim.Adam(params)

    # 训练
    best_val_metric = -1 if args.task_type=='cls' else 1e18
    best_epoch=-1; no_improve=0
    tag='+'.join(args.fusion_embed_types)
    ckpt=os.path.join(args.save_dir, f'best_arcmol_arc_reg_{tag}_seed{args.seed}.pth')
    calib=os.path.join(args.save_dir, f'calib_{tag}_seed{args.seed}.json')
    os.makedirs(os.path.dirname(ckpt),exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        if epoch==args.freeze_epochs+1:
            print(f"[Stage switch] Unfreeze fusion & descriptor at epoch {epoch}")
            freeze_fusion_desc(False)

        run_loss=0.0
        for (emb_dict, d), y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            emb_dict={k:v.to(device) for k,v in emb_dict.items()}
            d=d.to(device); y=y.to(device)

            # 模态丢弃
            if args.moddrop_p>0 and model.training:
                for k in list(emb_dict.keys()):
                    if random.random()<args.moddrop_p:
                        emb_dict[k]=torch.zeros_like(emb_dict[k])

            opt.zero_grad()
            mscale=cosine_warmup(epoch,args.margin_warmup_epochs)

            if args.task_type == 'cls':
                # 分类路径：保持原逻辑
                ls, lt, _, sparse_reg = model.forward_logits((emb_dict,d), y=y, margin_scale=mscale)
                loss_t = ce_loss(lt,y,ls=args.label_smooth)
                loss_s = ce_loss(ls,y,ls=args.label_smooth)
                lkd    = kd_loss_cls(ls, lt.detach(), T=2.0)
                kd_lambda = args.kd_max_lambda * cosine_warmup(epoch, args.margin_warmup_epochs)
                loss = 0.5*loss_t + 0.5*loss_s + kd_lambda*lkd + sparse_reg
                # === 分类新增：Proxy 原型正交正则（不改前向/结构） ===
                if args.cls_proxy_ortho_lambda > 0.0:
                    P = nn.functional.normalize(model.student.proxies, dim=-1)  # [C,D]
                    G = P @ P.t()                                             # Gram
                    I = torch.eye(G.size(0), device=G.device)
                    loss_ortho = (G - I).pow(2).mean()
                    loss = loss + args.cls_proxy_ortho_lambda * loss_ortho

            else:
                # 回归主干：保持 teacher/student + KD + 真值监督
                zs, zt, _, sparse_reg = model.forward_logits((emb_dict,d), y=None, margin_scale=1.0)
                yf = y.view_as(zs).float()
                loss_t = nn.functional.mse_loss(zt, yf)
                loss_s = nn.functional.mse_loss(zs, yf)
                lkd    = kd_loss_reg(zs, zt)
                kd_lambda = args.kd_max_lambda * cosine_warmup(epoch, args.margin_warmup_epochs)
                loss = 0.5*loss_t + 0.5*loss_s + kd_lambda*lkd + sparse_reg

                # === 仅回归新增：Arc 表征正则（不改变预测头） ===
                if model.arc_reg is not None and args.arc_reg_lambda>0:
                    # 取当前 batch 的融合表示并计算 Arc CE
                    x, _ = model.encode(emb_dict, d)
                    arc_ce, _ = model.arc_reg(x, y.view(-1,1).float(),
                                              margin_scale=mscale,
                                              class_balance=arc_class_balance)
                    loss = loss + args.arc_reg_lambda * arc_ce

            loss.backward()
            if args.max_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            run_loss += float(loss.item())

        # 验证（分类=教师logits；回归=教师回归）
        metrics, y_val, p_val, logits_val = eval_epoch(model, valid_loader, device,
                                                       task_type=args.task_type,
                                                       temperature=None, mc_passes=args.mc_passes,
                                                       y_std_enable=(args.task_type=='reg' and args.y_std_enable), y_mu=y_mu, y_sigma=y_sigma)

        if args.task_type == 'cls':
            auc, f1 = metrics['AUC'], metrics['F1']
            print(f"Epoch [{epoch}/{args.epochs}] TrainLoss {run_loss/len(train_loader):.4f} | ValAUC {auc:.4f} | F1 {f1:.4f}")
            cur_metric = auc
            better = (cur_metric > best_val_metric)
            if better:
                best_val_metric = cur_metric; best_epoch=epoch; no_improve=0
                torch.save(model.state_dict(), ckpt)
                if logits_val is not None:
                    T = temp_scaling_on_logits(y_val, logits_val)
                    prob_T = torch.softmax(torch.from_numpy(logits_val)/T,dim=1)[:,1].numpy()
                    thr = choose_prob_thr(y_val, prob_T)
                else:
                    T, thr = 1.0, 0.5
                save_calib(calib, T, thr)
                print(f"  -> Saved best model. Calibrated T={T:.3f}, prob threshold={thr:.3f}")
            else:
                no_improve+=1

        else:
            rmse, mae = metrics['RMSE'], metrics['MAE']
            print(f"Epoch [{epoch}/{args.epochs}] TrainLoss {run_loss/len(train_loader):.4f} | ValRMSE {rmse:.4f} | MAE {mae:.4f}")
            cur_metric = rmse
            better = (cur_metric < best_val_metric)
            if better:
                best_val_metric = cur_metric; best_epoch=epoch; no_improve=0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve+=1

        if no_improve>=args.patience:
            print("Early stopping.")
            break

    # ------- Eval helpers -------
    @torch.no_grad()
    def load_and_eval(name, loader):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(ckpt,map_location=device)); model.eval()

        if args.task_type == 'cls':
            T,thr = (1.0,0.5)
            if os.path.exists(calib):
                with open(calib,'r') as f: d=json.load(f)
                T=float(d.get('temperature',1.0)); thr=float(d.get('prob_threshold',0.5))

            metrics, y, p, _ = eval_epoch(model, loader, device, task_type='cls', temperature=T, mc_passes=args.mc_passes)
            cov=(p>thr).mean()
            acc = (((p>thr).astype(int)==y) * (p>thr)).sum() / max((p>thr).sum(),1)
            print(f"\\n=== {name} ===")
            print(f"{name:5s} AUC: {metrics['AUC']:.3f} | F1: {f1_score(y,(p>thr).astype(int)):.3f} | Coverage={cov:.3f}, Acc@accepted={acc:.3f} (prob_thr={thr:.2f})")
            return metrics['AUC']
        else:
            metrics, y, p, _ = eval_epoch(model, loader, device, task_type='reg', temperature=None, mc_passes=args.mc_passes,
                                         y_std_enable=(args.task_type=='reg' and args.y_std_enable), y_mu=y_mu, y_sigma=y_sigma)
            print(f"\\n=== {name} ===")
            print(f"{name:5s} RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
            return metrics['RMSE']

    test_metric  = load_and_eval("TEST",  test_loader)
    valid_metric = load_and_eval("VALID", valid_loader)
    _            = load_and_eval("TRAIN", train_loader)

    return ResultPack(best_val=best_val_metric, best_test=test_metric,
                      best_epoch=best_epoch, best_model_path=ckpt)


# -------------------------- main --------------------------
def main():
    parser=build_arg_parser()
    args=parser.parse_args()
    _ = train_and_eval(args)

if __name__=="__main__":
    main()
