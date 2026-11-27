"""MolBridge 的原子级联合图与多尺度图细化模型。"""

import math

import torch
from torch import Tensor, nn

from .graph import MoleculeBatch, MoleculeGraph


def assemble_pair(drug1: MoleculeBatch, drug2: MoleculeBatch) -> tuple[Tensor, Tensor, Tensor]:
    """在每个药对内部将两种药紧邻摆放，只在药对末尾补齐。"""
    if len(drug1.sizes) != len(drug2.sizes):
        raise ValueError("药对两侧的分子数量必须相同")
    pair_sizes = [a + b for a, b in zip(drug1.sizes, drug2.sizes)]
    batch_size, width = len(pair_sizes), max(pair_sizes)
    features = drug1.features.new_zeros((batch_size, width, drug1.features.size(-1)))
    bonds = drug1.bonds.new_zeros((batch_size, width, width))

    # 两个分子的化学键分别放在对角块，跨分子位置留给后续的 Q/K 建图。
    first_width = drug1.features.size(1)
    features[:, :first_width] = drug1.features
    bonds[:, :first_width, :first_width] = drug1.bonds
    for i, (start, size) in enumerate(zip(drug1.sizes, drug2.sizes)):
        end = start + size
        features[i, start:end] = drug2.features[i, :size]
        bonds[i, start:end, start:end] = drug2.bonds[i, :size, :size]

    lengths = torch.tensor(pair_sizes, device=features.device)
    padding = torch.arange(width, device=features.device)[None, :] >= lengths[:, None]
    return features, bonds, padding


class GraphRefinementBlock(nn.Module):
    """在同一张联合图上传递消息，再用前馈网络更新每个原子。"""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, atoms: Tensor, adjacency: Tensor) -> Tensor:
        messages = adjacency @ atoms
        atoms = self.message_norm(atoms + messages)
        return self.ffn_norm(atoms + self.ffn(atoms))


class MolBridge(nn.Module):
    """预测每个药对的事件分数，输出形状为 [药对数, 类别数]。"""

    def __init__(
        self,
        num_classes: int,
        atom_dim: int = 75,
        hidden_dim: int = 256,
        ffn_dim: int = 2048,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        classifier_dim: int = 256,
    ):
        super().__init__()
        if num_classes < 1 or num_layers < 1:
            raise ValueError("类别数和图细化层数必须大于零")
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError("隐藏维度和头数须为正整数，且隐藏维度须能被头数整除")
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_shape = (num_heads, hidden_dim // num_heads)
        self.edge_qk = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.edge_dropout = nn.Dropout(dropout)
        # 原注意力对合并的 Q/K/V 权重做 Xavier 初始化，保留其中 Q、K 的取值范围。
        bound = math.sqrt(6 / (hidden_dim + 3 * hidden_dim))
        nn.init.uniform_(self.edge_qk.weight, -bound, bound)
        nn.init.zeros_(self.edge_qk.bias)
        # 原模型用一个标量混合化学键图和学习图，各层共享同一张图。
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.blocks = nn.ModuleList(
            GraphRefinementBlock(hidden_dim, ffn_dim, dropout) for _ in range(num_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_dim),
            nn.ReLU(),
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def joint_adjacency(self, atoms: Tensor, bonds: Tensor, padding_mask: Tensor) -> Tensor:
        """在药对的全部有效原子上学习连边，再与含自环的化学键图混合。"""
        # 保留原建图方式：打分器独立学习，原子编码器的梯度来自后续消息传递。
        query, key = self.edge_qk(atoms.detach()).chunk(2, dim=-1)
        # 每个头独立比较原子，形状为 [药对数, 头数, 原子数, 每头维度]。
        query = query.unflatten(-1, self.head_shape).transpose(1, 2)
        key = key.unflatten(-1, self.head_shape).transpose(1, 2)
        scores = (query / math.sqrt(query.size(-1))) @ key.transpose(-2, -1)
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        # 对有效目标原子归一化，每个头单独做 dropout，再平均得到一张学习图。
        per_head_weights = scores.softmax(dim=-1)
        per_head_weights = self.edge_dropout(per_head_weights)
        edge_weights = per_head_weights.mean(dim=1)

        mix = self.graph_mix_logit.sigmoid()
        adjacency = (1 - mix) * bonds + mix * edge_weights
        padded_edges = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        return adjacency.masked_fill(padded_edges, 0)

    def forward(
        self, drug1: MoleculeGraph | MoleculeBatch, drug2: MoleculeGraph | MoleculeBatch
    ) -> Tensor:
        """输入单个药对或两列一一对应的分子，返回 [药对数, 类别数] 的分数。"""
        if isinstance(drug1, MoleculeGraph):
            drug1 = MoleculeBatch.from_graphs([drug1])
        if isinstance(drug2, MoleculeGraph):
            drug2 = MoleculeBatch.from_graphs([drug2])
        atom_features, bonds, padding_mask = assemble_pair(drug1, drug2)
        atoms = self.atom_encoder(atom_features)
        adjacency = self.joint_adjacency(atoms, bonds, padding_mask)

        # 读出同时使用初始特征与每一层的输出，保留不同传播深度的信息。
        multiscale = atoms
        for block in self.blocks:
            atoms = block(atoms, adjacency)
            multiscale = multiscale + atoms
        pairs = multiscale.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
        return self.classifier(pairs)
