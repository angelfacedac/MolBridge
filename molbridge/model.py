"""MolBridge 的原子级联合图与多尺度图细化模型。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
        atoms = self.message_norm(atoms + torch.bmm(adjacency, atoms))
        return self.ffn_norm(atoms + self.ffn(atoms))


class PairBatchNorm(nn.BatchNorm1d):
    """训练批次只有一个药对时，沿用运行统计量，以保留最后一个样本。"""

    def forward(self, pairs: Tensor) -> Tensor:
        if self.training and pairs.size(0) == 1:
            return F.batch_norm(
                pairs, self.running_mean, self.running_var,
                self.weight, self.bias, training=False, eps=self.eps,
            )
        return super().forward(pairs)


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
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        # 原模型用一个标量混合化学键图和学习图，各层共享同一张图。
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.blocks = nn.ModuleList(
            GraphRefinementBlock(hidden_dim, ffn_dim, dropout) for _ in range(num_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_dim),
            nn.ReLU(),
            PairBatchNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def joint_adjacency(self, atoms: Tensor, bonds: Tensor, padding_mask: Tensor) -> Tensor:
        """在药对的全部有效原子上学习连边，再与含自环的化学键图混合。"""
        # 保留原建图方式：打分器独立学习，原子编码器的梯度来自后续消息传递。
        graph_inputs = atoms.detach()
        _, learned = self.attention(
            graph_inputs, graph_inputs, graph_inputs,
            key_padding_mask=padding_mask, average_attn_weights=True,
        )
        mix = self.graph_mix_logit.sigmoid()
        adjacency = (1 - mix) * bonds + mix * learned
        padded_edges = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        return adjacency.masked_fill(padded_edges, 0)

    def forward(self, atom_features: Tensor, bonds: Tensor, padding_mask: Tensor) -> Tensor:
        """输入为配对原子特征 [B,N,F]、化学键图 [B,N,N] 和填充标记 [B,N]。

        N 为批内补齐后的联合原子数。两种药的化学键各占一个对角块。
        padding_mask 为布尔张量，True 表示填充位置。
        """
        atoms = self.atom_encoder(atom_features)
        adjacency = self.joint_adjacency(atoms, bonds, padding_mask)

        # 读出同时使用初始特征与每一层的输出，保留不同传播深度的信息。
        multiscale = atoms
        for block in self.blocks:
            atoms = block(atoms, adjacency)
            multiscale = multiscale + atoms
        pairs = multiscale.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
        return self.classifier(pairs)
