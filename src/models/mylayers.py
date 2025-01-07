import torch
from torch import nn

from src.load_config import CONFIG

BATCH_SIZE = CONFIG['train']['batch_size']
NUM_CLASSES = CONFIG['data']['num_classes']
ATOM_DIM = CONFIG['data']['atom_dim']
MLP_HIDDEN_DIM = CONFIG['model']['mlp']['hidden_dim']
BLOCK_FFN_HIDDEN_DIM = CONFIG['model']['block']['ffn']['hidden_dim']
DROPOUT_P = CONFIG['train']['dropout']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']


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


class ReconstructGraph(nn.Module):
    def __init__(self):
        super(ReconstructGraph, self).__init__()

        self.w_q = nn.Parameter(torch.zeros(ATOM_DIM, ATOM_DIM // 2))
        self.w_k = nn.Parameter(torch.zeros(ATOM_DIM, ATOM_DIM // 2))
        self.bias = nn.Parameter(torch.zeros(ATOM_DIM // 2))
        self.a = nn.Parameter(torch.zeros(ATOM_DIM // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant, adj_mask=None):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        # values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        if adj_mask is not None:
            attentions = torch.bmm(adj_mask, attentions)
            attentions = torch.bmm(attentions, adj_mask)
        return attentions

