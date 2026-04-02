#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# ================== 日志和 RDKit 提示全关 ==================
import os
import logging
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger("ogb").setLevel(logging.ERROR)
logging.getLogger("ogb").propagate = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")   # 关掉 [xx] DEPRECATION WARNING

# ================== imports ==================
from typing import List, Dict
import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.Chem.AtomPairs import Pairs, Torsions  # 你原来项目用的写法

# 你项目里的本地模型
from models.MolT5.utils import init_molt5, molt5_embedding
from models.BioT5.utils import init_biot5, biot5_embedding
from models.ChemBERTa.utils import init_chemberta, chemberta_embedding
from models.MolCLR.utils import init_molclr, molclr_embedding
from models.SSL.utils import init_ssl, ssl_embedding
from models.UniMolV1.utils import init_unimol1, unimol1_embedding
from models.UniMolV2.utils import init_unimol2, unimol2_embedding


# ======================================================
# 工具：bitvect -> torch
# ======================================================
def fp_to_tensor(fp) -> torch.Tensor:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr).float()


# ======================================================
# 兼容各版本 RDKit 的 Morgan 获取函数
# ======================================================
def morgan_bitvect(
    mol: Chem.Mol,
    radius: int,
    nBits: int = 2048,
    useFeatures: bool = False,
    useChirality: bool = False,
):
    """
    先试新版 rdFingerprintGenerator（不会报弃用），
    不行就退回老的 AllChem.GetMorganFingerprintAsBitVect（会被我们关日志）。
    """
    # 1) 先试 generator
    try:
        # 老版本的签名比较短，不能传一堆 kw，只传它能吃的
        gen = rdFPGen.GetMorganGenerator(
            radius=radius,
            includeChirality=useChirality,
            fpSize=nBits,
        )
        # 旧版 generator 没有 useFeatures 这一说，想要 feature 版只好走老接口
        if useFeatures:
            raise RuntimeError("generator in this RDKit has no useFeatures")
        fp = gen.GetFingerprint(mol)
        return fp
    except Exception:
        # 2) 回退老接口（就是你最初看到的那个）
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=nBits,
            useChirality=useChirality,
            useFeatures=useFeatures,
        )
        return fp


# ======================================================
# 1. 注册表
# ======================================================
class EmbeddingRegistry:
    _registry: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, batch_support: bool = True):
        def deco(klass):
            cls._registry[name] = {"class": klass, "batch_support": batch_support}
            return klass
        return deco

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"embedding '{name}' not registered")
        klass = cls._registry[name]["class"]
        try:
            return klass(**kwargs)
        except TypeError:
            return klass()

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._registry.keys())


# ======================================================
# 2. 基类
# ======================================================
class BaseEmbedder:
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        raise NotImplementedError

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, mols, batch_size: int = 64) -> torch.Tensor:
        if isinstance(mols, (Chem.Mol, str)):
            mols = [mols]
        fixed: List[Chem.Mol] = []
        for it in mols:
            if isinstance(it, Chem.Mol):
                fixed.append(it)
            elif isinstance(it, str):
                m = Chem.MolFromSmiles(it)
                if m is None:
                    raise ValueError(f"invalid SMILES: {it}")
                fixed.append(m)
            else:
                raise TypeError("input must be Mol/SMILES/list")
        return self.embed_batch(fixed)

# ======================================================
# 3. 描述符（含你说的那两个）
# ======================================================
# ---- RDKit Descriptor 全量表征 ----
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

@EmbeddingRegistry.register("RDKitDescriptors", batch_support=True)
class RDKitDescriptorEmbedder(BaseEmbedder):
    def __init__(self):
        # RDKit 自带的全量描述符
        self.desc_names = [x[0] for x in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.desc_names)

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        vals = self.calc.CalcDescriptors(mol)
        return torch.tensor(vals, dtype=torch.float32)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


# ---- Mordred 描述符（可选）----
try:
    from mordred import Calculator, descriptors as mordred_descriptors

    @EmbeddingRegistry.register("Mordred", batch_support=True)
    class MordredEmbedder(BaseEmbedder):
        def __init__(self):
            # 一次性建好，会很全
            self.calc = Calculator(mordred_descriptors, ignore_3D=True)

        def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
            res = self.calc(mol)
            # 有些字段是 None，需要转成 0 或 NaN
            vals = []
            for v in res:
                if v is None:
                    vals.append(0.0)
                else:
                    vals.append(float(v))
            return torch.tensor(vals, dtype=torch.float32)

        def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
            return torch.stack([self.embed_single(m) for m in mols], dim=0)
except Exception:
    # 没装 mordred 就算了
    pass


# ======================================================
# 3. RDKit 指纹（含你说的那两个）
# ======================================================

@EmbeddingRegistry.register("RDKFingerprint", batch_support=True)
class RDKFingerprintEmbedder(BaseEmbedder):
    def __init__(self, fpSize: int = 2048, minPath: int = 1, maxPath: int = 7):
        self.fpSize = fpSize
        self.minPath = minPath
        self.maxPath = maxPath

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = Chem.RDKFingerprint(
            mol,
            fpSize=self.fpSize,
            minPath=self.minPath,
            maxPath=self.maxPath,
        )
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


@EmbeddingRegistry.register("MACCSkeys", batch_support=True)
class MACCSEmbedder(BaseEmbedder):
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


@EmbeddingRegistry.register("EStateFingerprint", batch_support=True)
class EStateEmbedder(BaseEmbedder):
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        est, _ = Fingerprinter.FingerprintMol(mol)
        return torch.tensor(est, dtype=torch.float32)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


# ---- Morgan / ECFP ----
@EmbeddingRegistry.register("MorganFingerprint", batch_support=True)
class MorganFingerprintEmbedder(BaseEmbedder):
    def __init__(self, radius: int = 2, nBits: int = 2048, useChirality: bool = False):
        self.radius = radius
        self.nBits = nBits
        self.useChirality = useChirality

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = morgan_bitvect(
            mol,
            radius=self.radius,
            nBits=self.nBits,
            useFeatures=False,
            useChirality=self.useChirality,
        )
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


@EmbeddingRegistry.register("ECFP2", batch_support=True)
class ECFP2Embedder(MorganFingerprintEmbedder):
    def __init__(self):
        super().__init__(radius=1, nBits=2048, useChirality=False)


@EmbeddingRegistry.register("ECFP4", batch_support=True)
class ECFP4Embedder(MorganFingerprintEmbedder):
    def __init__(self):
        super().__init__(radius=2, nBits=2048, useChirality=False)


@EmbeddingRegistry.register("ECFP6", batch_support=True)
class ECFP6Embedder(MorganFingerprintEmbedder):
    def __init__(self):
        super().__init__(radius=3, nBits=2048, useChirality=False)


# ---- FCFP（feature 版只能走老接口，所以也包进 morgan_bitvect）----
@EmbeddingRegistry.register("FCFP2", batch_support=True)
class FCFP2Embedder(BaseEmbedder):
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = morgan_bitvect(mol, radius=1, nBits=2048, useFeatures=True, useChirality=False)
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


@EmbeddingRegistry.register("FCFP4", batch_support=True)
class FCFP4Embedder(BaseEmbedder):
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = morgan_bitvect(mol, radius=2, nBits=2048, useFeatures=True, useChirality=False)
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


@EmbeddingRegistry.register("FCFP6", batch_support=True)
class FCFP6Embedder(BaseEmbedder):
    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = morgan_bitvect(mol, radius=3, nBits=2048, useFeatures=True, useChirality=False)
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)



# ---- Pattern ----
@EmbeddingRegistry.register("PatternFingerprint", batch_support=True)
class PatternFingerprintEmbedder(BaseEmbedder):
    def __init__(self, fpSize: int = 2048):
        self.fpSize = fpSize

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = Chem.PatternFingerprint(mol, fpSize=self.fpSize)
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


# ---- Layered ----
@EmbeddingRegistry.register("LayeredFingerprint", batch_support=True)
class LayeredFingerprintEmbedder(BaseEmbedder):
    def __init__(self, fpSize: int = 2048, minPath: int = 1, maxPath: int = 7):
        self.fpSize = fpSize
        self.minPath = minPath
        self.maxPath = maxPath

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        fp = Chem.LayeredFingerprint(
            mol,
            minPath=self.minPath,
            maxPath=self.maxPath,
            fpSize=self.fpSize,
        )
        return fp_to_tensor(fp)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        return torch.stack([self.embed_single(m) for m in mols], dim=0)


# ---- Avalon (可选) ----
try:
    from rdkit.Avalon import pyAvalonTools

    @EmbeddingRegistry.register("AvalonFingerprint", batch_support=True)
    class AvalonFingerprintEmbedder(BaseEmbedder):
        def __init__(self, nBits: int = 512, isQuery: bool = False):
            self.nBits = nBits
            self.isQuery = isQuery

        def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.nBits, isQuery=self.isQuery)
            arr = torch.zeros(self.nBits, dtype=torch.float32)
            onbits = [idx for idx in fp.GetOnBits() if idx < self.nBits]  # 过滤越界索引
            if onbits:
                arr[onbits] = 1.0
            return arr

        def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
            return torch.stack([self.embed_single(m) for m in mols], dim=0)
except ImportError:
    pass


# ---- Pharm2D (可选) ----
try:
    import os
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.Pharm2D import Generate
    from rdkit.Chem.Pharm2D.SigFactory import SigFactory

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(featFactory,
                            minPointCount=2,
                            maxPointCount=3,
                            trianglePruneBins=False)
    sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    sigFactory.Init()

    @EmbeddingRegistry.register("Pharm2D", batch_support=True)
    class Pharm2DEmbedder(BaseEmbedder):
        def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
            sig = Generate.Gen2DFingerprint(mol, sigFactory)
            arr = torch.zeros(sig.GetNumBits(), dtype=torch.float32)
            onbits = list(sig.GetOnBits())
            if onbits:
                arr[onbits] = 1.0
            return arr

        def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
            return torch.stack([self.embed_single(m) for m in mols], dim=0)
except Exception:
    pass


# ======================================================
# 4. 预训练模型抽象
# ======================================================
class BaseModelEmbedder(BaseEmbedder):
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self._init_model()

    def _init_model(self):
        raise NotImplementedError

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def embed_single(self, mol: Chem.Mol) -> torch.Tensor:
        smi = Chem.MolToSmiles(mol)
        out = self._embed_smiles_batch([smi])
        return out.squeeze(0)

    def embed_batch(self, mols: List[Chem.Mol]) -> torch.Tensor:
        smiles = [Chem.MolToSmiles(m) for m in mols]
        outs = []
        for i in range(0, len(smiles), self.batch_size):
            chunk = smiles[i:i + self.batch_size]
            emb = self._embed_smiles_batch(chunk)
            outs.append(emb)
        return torch.cat(outs, dim=0)


# ======================================================
# 5. 文本模型
# ======================================================
@EmbeddingRegistry.register("MolT5", batch_support=True)
class MolT5Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_molt5()

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        return molt5_embedding(smiles, self.tokenizer, self.model)


@EmbeddingRegistry.register("BioT5", batch_support=True)
class BioT5Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_biot5()

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        return biot5_embedding(smiles, self.tokenizer, self.model)


@EmbeddingRegistry.register("ChemBERTa", batch_support=True)
class ChemBERTaEmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_chemberta()

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        return chemberta_embedding(smiles, self.tokenizer, self.model)


# ======================================================
# 6. MolCLR
# ======================================================
@EmbeddingRegistry.register("MolCLR", batch_support=True)
class MolCLREmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_molclr()

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = molclr_embedding(smiles, self.tokenizer, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


# ======================================================
# 7. SSL 五个
# ======================================================
@EmbeddingRegistry.register("AttrMask", batch_support=True)
class AttrMaskEmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_ssl("AttrMask")

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = ssl_embedding(smiles, None, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


@EmbeddingRegistry.register("GPT-GNN", batch_support=True)
class GPTGNNEmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_ssl("GPT-GNN")

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = ssl_embedding(smiles, None, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


@EmbeddingRegistry.register("GraphCL", batch_support=True)
class GraphCLEmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_ssl("GraphCL")

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = ssl_embedding(smiles, None, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


@EmbeddingRegistry.register("GraphMVP", batch_support=True)
class GraphMVPEmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_ssl("GraphMVP")

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = ssl_embedding(smiles, None, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


@EmbeddingRegistry.register("GROVER", batch_support=True)
class GROVEREmbedder(BaseModelEmbedder):
    def _init_model(self):
        self.tokenizer, self.model = init_ssl("GROVER")

    def _embed_smiles_batch(self, smiles: List[str]) -> torch.Tensor:
        emb = ssl_embedding(smiles, None, self.model)
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb


# ======================================================
# 8. 3D Gemo
# ======================================================
@EmbeddingRegistry.register("UniMolV1", batch_support=True)
class UniMolV1Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.model, self.dim = init_unimol1(tag="UniMol1_noH", device="cuda")

    def _embed_smiles_batch(self, smiles):
        return unimol1_embedding(smiles, self.model)



@EmbeddingRegistry.register("UniMolV2_84M", batch_support=True)
class UniMol2_84M_Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.model, self.dim = init_unimol2("UniMol2_84M", device="cuda")
    def _embed_smiles_batch(self, smiles):
        return unimol2_embedding(smiles, self.model)

# @EmbeddingRegistry.register("UniMol2_164M", batch_support=True)
# class UniMol2_164M_Embedder(BaseModelEmbedder):
#     def _init_model(self):
#         self.model, self.dim = init_unimol2("UniMol2_164M", device="cuda")
#     def _embed_smiles_batch(self, smiles):
#         return unimol2_embedding(smiles, self.model)

@EmbeddingRegistry.register("UniMolV2_310M", batch_support=True)
class UniMol2_310M_Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.model, self.dim = init_unimol2("UniMol2_310M", device="cuda")
    def _embed_smiles_batch(self, smiles):
        return unimol2_embedding(smiles, self.model)

# @EmbeddingRegistry.register("UniMol2_570M", batch_support=True)
# class UniMol2_570M_Embedder(BaseModelEmbedder):
#     def _init_model(self):
#         self.model, self.dim = init_unimol2("UniMol2_570M", device="cuda")
#     def _embed_smiles_batch(self, smiles):
#         return unimol2_embedding(smiles, self.model)

@EmbeddingRegistry.register("UniMolV2_1.1B", batch_support=True)
class UniMol2_11B_Embedder(BaseModelEmbedder):
    def _init_model(self):
        self.model, self.dim = init_unimol2("UniMol2_1.1B", device="cuda")
    def _embed_smiles_batch(self, smiles):
        return unimol2_embedding(smiles, self.model)


# ======================================================
# 8. 对外
# ======================================================
class Embedding:
    @staticmethod
    def available() -> List[str]:
        return EmbeddingRegistry.names()

    @staticmethod
    def get(name: str, **kwargs) -> BaseEmbedder:
        return EmbeddingRegistry.create(name, **kwargs)


# ======================================================
# 9. demo
# ======================================================
if __name__ == "__main__":
    test_smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
    ]
    print("Available embeddings:", Embedding.available())
    for name in Embedding.available():
        print(f"\n#### embedding: {name}")
        emb = Embedding.get(name, batch_size=16)
        out = emb(test_smiles, batch_size=16)
        print(out.shape)
