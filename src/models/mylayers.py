import torch
from torch import nn

from src.load_config import CONFIG

NUM_CLASSES = CONFIG['data']['num_classes']
ATOM_DIM = CONFIG['data']['atom_dim']
MLP_HIDDEN_DIM = CONFIG['model']['mlp']['hidden_dim']
BLOCK_FFN_HIDDEN_DIM = CONFIG['model']['block']['ffn']['hidden_dim']
DROPOUT_P = CONFIG['train']['dropout']


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ATOM_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(MLP_HIDDEN_DIM, NUM_CLASSES)
        )

    def forward(self, embeds):
        return self.net(embeds)


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(ATOM_DIM, BLOCK_FFN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(BLOCK_FFN_HIDDEN_DIM, ATOM_DIM),
            nn.Dropout(p=DROPOUT_P)
        )
        self.norm1 = nn.LayerNorm(ATOM_DIM)
        self.norm2 = nn.LayerNorm(ATOM_DIM)

    def messaging(self, adjs, embeds):
        return torch.bmm(adjs, embeds)  # 批量矩阵乘法，只适用于稠密矩阵

    def forward(self, adjs, embeds):
        embeds = self.norm1(embeds + self.messaging(adjs, embeds))
        embeds = self.norm2(embeds + self.ffn(embeds))
        return embeds