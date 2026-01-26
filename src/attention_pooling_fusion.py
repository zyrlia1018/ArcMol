import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def _batch_centered_gram_rbf(X: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    RBF Gram matrix for a batch of vectors.
    X: [B, D] -> K: [B, B] with K[i,j] = exp(-gamma * ||x_i - x_j||^2)
    """
    sq_norms = (X * X).sum(dim=1, keepdim=True)
    dist2 = sq_norms + sq_norms.t() - 2.0 * (X @ X.t())
    K = torch.exp(-gamma * torch.clamp(dist2, min=0.0))
    return K


def _cka_rbf(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1e-3, eps: float = 1e-8) -> torch.Tensor:
    """
    Biased RBF-CKA across the batch (a common practical variant).
    X, Y: [B, D] -> scalar in [0,1]
    """
    B = X.size(0)
    device = X.device
    H = torch.eye(B, device=device) - (1.0 / B) * torch.ones((B, B), device=device)
    Kx = _batch_centered_gram_rbf(X, gamma)
    Ky = _batch_centered_gram_rbf(Y, gamma)
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H
    hsic_xy = torch.sum(Kx_c * Ky_c)
    hsic_xx = torch.sum(Kx_c * Kx_c)
    hsic_yy = torch.sum(Ky_c * Ky_c)
    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
    return torch.clamp(hsic_xy / (denom + eps), min=0.0, max=1.0)


class AttentionPoolingFusion(nn.Module):
    """
    Attention pooling with complementarity bias (Cosine or RBF-CKA) and optional task-aware gating.
    新增：MoE gating (top-k) + 稀疏熵正则（可选）。
    forward 返回: (fused, sparse_reg)
    """

    def __init__(
        self,
        used_embedding_types: List[str],
        l_output_dim: int = 128,
        hidden_dim: int = 64,
        dropout_prob: float = 0.5,
        comp_mode: str = 'cos',           # 'cos' | 'cka_rbf'
        cka_gamma: float = 1e-3,
        task_gate: str = 'scalar',        # 'none' | 'scalar' | 'mlp'
        task_ctx_dim: int = 16,
        comp_scale: float = 1.0,
        top_k: Optional[int] = None,      # MoE：每次只选 top-k 个分支
        sparse_lambda: float = 0.0        # 稀疏熵正则系数（>=0）
    ):
        super().__init__()
        self.used_embedding_types = list(used_embedding_types)
        self.comp_mode = comp_mode
        self.cka_gamma = float(cka_gamma)
        self.task_gate = task_gate
        self.task_ctx_dim = task_ctx_dim
        self.top_k = top_k
        self.sparse_lambda = float(sparse_lambda)

        # 你的各 embedding 维度表
        self.embedding_dim_dict = {
            "RDKFingerprint": 2048, "MACCSkeys": 167, "EStateFingerprint": 79,
            "MolT5": 768, "BioT5": 768, "AttrMask": 300, "GPT-GNN": 300,
            "GraphCL": 300, "MolCLR": 512, "GraphMVP": 300, "GROVER": 300,
            "UniMolV1": 512, "UniMolV2_84M": 768, "UniMolV2_1.1B": 1536
        }
        self.embedding_to_128_layers = nn.ModuleList([
            nn.Linear(self.embedding_dim_dict.get(embed_type, 512), l_output_dim)
            for embed_type in self.used_embedding_types
        ])

        self.attention_mlp = nn.Sequential(
            nn.Linear(l_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 可学习的互补性缩放
        self.comp_beta = nn.Parameter(torch.tensor(float(comp_scale)))

        # 任务门控
        if self.task_gate == 'scalar':
            self.task_gate_layer = nn.Linear(task_ctx_dim, len(self.used_embedding_types))
        elif self.task_gate == 'mlp':
            self.task_gate_layer = nn.Sequential(
                nn.Linear(task_ctx_dim, max(16, task_ctx_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(16, task_ctx_dim // 2), len(self.used_embedding_types))
            )
        else:
            self.task_gate_layer = None

        self.dropout = nn.Dropout(dropout_prob)

    @staticmethod
    def _mask_from_transformed(t: torch.Tensor) -> torch.Tensor:
        # t: [B, D] -> [B,1,1] 有效掩码
        valid = (~torch.isnan(t).any(dim=-1) & (t.norm(dim=-1) > 1e-8)).float()
        return valid.view(t.size(0), 1, 1)

    def _complementarity_vector(self, branches: List[torch.Tensor]) -> torch.Tensor:
        """
        branches: N * [B,D]
        returns: comp_vec [B,N,1]
        """
        B = branches[0].size(0)
        N = len(branches)

        if self.comp_mode == 'cos':
            normed = [F.normalize(x, dim=-1) for x in branches]
            S = torch.stack(normed, dim=0).transpose(0, 1)  # [B,N,D]
            cos_sim = torch.einsum('bnd,bmd->bnm', S, S)   # [B,N,N]
            comp_mat = 1.0 - cos_sim
            comp_vec = comp_mat.mean(dim=-1, keepdim=True)  # [B,N,1]
            return comp_vec

        elif self.comp_mode == 'cka_rbf':
            device = branches[0].device
            cka_mat = torch.zeros((N, N), device=device)
            for i in range(N):
                for j in range(N):
                    cka_mat[i, j] = _cka_rbf(branches[i], branches[j], gamma=self.cka_gamma)
            comp_mat = 1.0 - cka_mat                    # [N,N]
            comp_mat = comp_mat.unsqueeze(0).expand(B, -1, -1)  # [B,N,N]
            comp_vec = comp_mat.mean(dim=-1, keepdim=True)      # [B,N,1]
            return comp_vec

        else:
            raise ValueError(f"Unknown comp_mode: {self.comp_mode}")

    def forward(self, data_dict: Dict[str, torch.Tensor], task_context: Optional[torch.Tensor] = None):
        # 1) 统一投影
        all_proj: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for idx, embed_type in enumerate(self.used_embedding_types):
            if embed_type not in data_dict:
                raise ValueError(f"Missing {embed_type} in input data")
            x = data_dict[embed_type].to(dtype=torch.float32)
            if x.dim() > 2: x = x.squeeze()
            if x.dim() == 1: x = x.unsqueeze(0)
            proj = self.embedding_to_128_layers[idx](x)  # [B,D]
            all_proj.append(proj)
            mask_list.append(self._mask_from_transformed(proj))

        embeds = torch.stack(all_proj, dim=1)  # [B,N,D]
        mask = torch.cat(mask_list, dim=1)     # [B,N,1]

        # 2) 内容注意力 + 互补性偏置 + 任务门控
        attn_scores = self.attention_mlp(embeds)             # [B,N,1]
        comp_vec = self._complementarity_vector(all_proj)    # [B,N,1]
        attn_scores = attn_scores + self.comp_beta * comp_vec

        if self.task_gate_layer is not None and task_context is not None:
            task_bias_raw = self.task_gate_layer(task_context)   # [B,N]
            task_bias = task_bias_raw.unsqueeze(-1)              # [B,N,1]
            attn_scores = attn_scores + task_bias

        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(-1)
            mask = mask.expand_as(attn_scores)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)  # [B,N,1]

        # 3) MoE top-k 门控（按注意力强度选前 K 个分支）
        if self.top_k is not None and self.top_k < attn_weights.size(1):
            topk_vals, topk_idx = torch.topk(attn_weights.squeeze(-1), self.top_k, dim=1)
            sel_mask = torch.zeros_like(attn_weights.squeeze(-1))
            sel_mask.scatter_(1, topk_idx, 1.0)
            sel_mask = sel_mask.unsqueeze(-1)             # [B,N,1]
            attn_weights = attn_weights * sel_mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 4) 融合
        fused = torch.sum(attn_weights * embeds, dim=1)    # [B,D]

        # 5) 稀疏熵正则（越小越稀疏）
        sparse_reg = 0.0
        if self.sparse_lambda > 0:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
            sparse_reg = self.sparse_lambda * entropy

        return self.dropout(fused), sparse_reg
