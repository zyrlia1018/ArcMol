import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_ROOT = os.environ.get("DEEPFP_ASSETS_DIR", os.path.join(_THIS_DIR, "assets"))

CHEMBERTA_PATH = os.path.join(ASSETS_ROOT, "ChemBERTa-77M-MTR")
BIOT5_PATH = os.path.join(ASSETS_ROOT, "BioT5")
MOLT5_PATH = os.path.join(ASSETS_ROOT, "MolT5")
CACHE_PATH = os.path.join(ASSETS_ROOT, "cache.json")
MOLCLR_PATH = os.path.join(ASSETS_ROOT, "MolCLR", "model.pth")
MOTIF_PATH = os.path.join(ASSETS_ROOT, "Motif", "model.pth")
ATTRMASK_PATH = os.path.join(ASSETS_ROOT, "AM", "model.pth")
GPTGNN_PATH = os.path.join(ASSETS_ROOT, "GPT_GNN", "model.pth")
GRAPHCL_PATH = os.path.join(ASSETS_ROOT, "GraphCL", "model.pth")
GRAPHMVP_PATH = os.path.join(ASSETS_ROOT, "GraphMVP", "model.pth")

# ==== UniMol V2 (local) ====
UNIMOL2_ROOT = os.path.join(ASSETS_ROOT, "UniMolV2")
UNIMOL2_84M_PATH = os.path.join(UNIMOL2_ROOT, "84M", "checkpoint.pt")
UNIMOL2_164M_PATH = os.path.join(UNIMOL2_ROOT, "164M", "checkpoint.pt")
UNIMOL2_310M_PATH = os.path.join(UNIMOL2_ROOT, "310M", "checkpoint.pt")
# 历史目录可能叫 580M，这里做兼容。
_u2_570 = os.path.join(UNIMOL2_ROOT, "570M", "checkpoint.pt")
_u2_580 = os.path.join(UNIMOL2_ROOT, "580M", "checkpoint.pt")
UNIMOL2_570M_PATH = _u2_570 if os.path.exists(_u2_570) else _u2_580
UNIMOL2_1_1B_PATH = os.path.join(UNIMOL2_ROOT, "1.1B", "checkpoint.pt")

# ==== UniMol V1 (local) ====
UNIMOL1_ROOT = os.path.join(ASSETS_ROOT, "UniMolV1")
UNIMOL1_NO_H_PATH = os.path.join(UNIMOL1_ROOT, "mol_pre_no_h_220816.pt")
UNIMOL1_ALL_H_PATH = os.path.join(UNIMOL1_ROOT, "mol_pre_all_h_220816.pt")
UNIMOL1_DICT_PATH = os.path.join(UNIMOL1_ROOT, "mol.dict.txt")

BLOCKSIZE = 1e6
