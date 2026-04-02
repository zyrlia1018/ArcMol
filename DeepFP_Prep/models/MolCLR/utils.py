import os
import sys
import yaml
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem

# 如果你这个文件本来就是包内相对导入，就保留这两行
from .dataset import MoleculeDataset, ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST
from .model import GINet
from utils.env_utils import *   # 有 MOLCLR_PATH 等 :contentReference[oaicite:1]{index=1}

embedType = 'MolCLR'
batch_size = 1000
chunksize = 1e6

def init_molclr():
    model = GINet(
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0,
        pool='mean',
    )
    model.load_state_dict(torch.load(MOLCLR_PATH, map_location='cpu'))
    model.eval()
    model = model.cpu()
    return None, model

# 把“一个 smiles → Data”的那段从原文件单条逻辑里拆出来
def _smiles_to_data(smi: str) -> Data:
    if len(smi) >= 600:
        mol = Chem.MolFromSmiles('COO')
    else:
        mol = Chem.MolFromSmiles(smi)

    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum() if atom.GetAtomicNum() != 0 else 1))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

@torch.no_grad()
def molclr_embedding(smiles, place_holder, model):
    # 单条：保持你原来的行为
    if isinstance(smiles, str):
        data = _smiles_to_data(smiles)
        return model(data)

    # 多条：自己拼成 Batch
    if isinstance(smiles, list):
        data_list = [_smiles_to_data(s) for s in smiles]
        batch = Batch.from_data_list(data_list)
        return model(batch)

    raise ValueError("smiles must be str or list[str]")

@torch.no_grad()
def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    dataset = MoleculeDataset(SMILES_LIB_PATH)

    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=25,
                            drop_last=False,
                            shuffle=False)

    model = GINet(
        num_layer=config['model']['num_layer'],
        emb_dim=config['model']['emb_dim'],
        feat_dim=config['model']['feat_dim'],
        drop_ratio=config['model']['drop_ratio'],
        pool=config['model']['pool'],
    )
    model.load_state_dict(torch.load(MOLCLR_PATH, map_location='cpu'))
    model.eval()
    model = model.cuda()

    os.makedirs(os.path.join(EMBEDDING_DIR, embedType), exist_ok=True)

    embedding_list = []
    for i, data in enumerate(dataloader):
        data = data.cuda()
        emb = model(data)
        embedding_list.append(emb.cpu())

        if (i + 1) * batch_size % chunksize == 0:
            final_embedding = torch.cat(embedding_list, dim=0)
            torch.save(final_embedding,
                       os.path.join(EMBEDDING_DIR, embedType, f"{(i + 1) * batch_size // chunksize}.pt"))
            embedding_list = []

    final_embedding = torch.cat(embedding_list, dim=0)
    torch.save(final_embedding, os.path.join(EMBEDDING_DIR, embedType, "last.pt"))
