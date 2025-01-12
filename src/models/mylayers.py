import torch
from torch import nn
from torch_sparse import SparseTensor

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


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

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


class GRUPYG(nn.Module):
    def __init__(self):
        super(GRUPYG, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(ATOM_DIM, BLOCK_FFN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(BLOCK_FFN_HIDDEN_DIM, ATOM_DIM),
            nn.Dropout(p=DROPOUT_P)
        )
        self.norm1 = nn.LayerNorm(ATOM_DIM)
        self.norm2 = nn.LayerNorm(ATOM_DIM)

    def messaging(self, adj_coo, embeds):
        row, col = adj_coo
        adj_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=torch.ones(row.size(), dtype=torch.float32),
            size=(embeds.size(0), embeds.size(0)),
            device=adj_coo.device
        )
        # 使用 torch.sparse.mm 进行稀疏矩阵和稠密矩阵的乘法
        return torch.sparse.mm(adj_sparse, embeds)
        # return SparseTensor(
        #     row=adj_coo[0], col=adj_coo[1], sparse_sizes=(embeds.size(0), embeds.size(0))
        # ).to_device(adj_coo.device) @ embeds

    def forward(self, adj_coo, embeds):
        embeds = self.norm1(embeds + self.messaging(adj_coo, embeds))
        embeds = self.norm2(embeds + self.ffn(embeds))
        return embeds


class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()

        self.w_q = nn.Parameter(torch.zeros(ATOM_DIM, ATOM_DIM // 2))
        self.w_k = nn.Parameter(torch.zeros(ATOM_DIM, ATOM_DIM // 2))
        self.bias = nn.Parameter(torch.zeros(ATOM_DIM // 2))
        self.a = nn.Parameter(torch.zeros(ATOM_DIM // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant, mask=None):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        # values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        if mask is not None:
            attentions = attentions.masked_fill(mask, -1e9)
        return attentions

