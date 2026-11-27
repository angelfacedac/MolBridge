"""数据层与模型共用的单分子图和分子批次。"""

from typing import NamedTuple, Sequence

import torch
from torch import Tensor


ATOM_DIM = 75


class MoleculeGraph(NamedTuple):
    """一个分子的原子特征 [N,F] 和化学键邻接矩阵 [N,N]。"""

    features: Tensor
    bonds: Tensor


class MoleculeBatch(NamedTuple):
    """一列独立分子，按该列最长分子补齐，sizes 保存各分子的真实原子数。"""

    features: Tensor
    bonds: Tensor
    sizes: tuple[int, ...]

    @classmethod
    def from_graphs(cls, graphs: Sequence[MoleculeGraph]) -> "MoleculeBatch":
        sizes = tuple(len(graph.features) for graph in graphs)
        width = max(sizes)
        features = graphs[0].features.new_zeros((len(graphs), width, graphs[0].features.size(-1)))
        bonds = graphs[0].bonds.new_zeros((len(graphs), width, width))
        for i, (graph, size) in enumerate(zip(graphs, sizes)):
            features[i, :size] = graph.features
            bonds[i, :size, :size] = graph.bonds
        return cls(features, bonds, sizes)

    def to(self, device: torch.device) -> "MoleculeBatch":
        return MoleculeBatch(self.features.to(device), self.bonds.to(device), self.sizes)


def molecule_graph(smiles: str) -> MoleculeGraph:
    """生成全部原子的特征和带自环的化学键图。"""
    # 只有从 SMILES 构图时才需要化学工具，模型可以直接接收已有的张量。
    from rdkit import Chem
    from .features import atom_features

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("无法解析为含原子的分子")
    # 与公开代码保持相同的规范化、编号顺序，便于用旧权重核对编码。
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    if mol is None:
        raise ValueError("规范化后的分子无法解析")
    mol = Chem.RenumberAtoms(mol, list(Chem.CanonicalRankAtoms(mol)))
    features = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    bonds = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.float32)
    bonds.fill_diagonal_(1)
    return MoleculeGraph(features, bonds)
