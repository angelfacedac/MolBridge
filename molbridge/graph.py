"""数据层与模型共用的单分子图和分子批次。"""

from typing import NamedTuple, Sequence

import torch
from torch import Tensor


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
