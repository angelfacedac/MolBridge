import torch
from torch import nn
from torch_sparse import SparseTensor
import torch.nn.functional as F

from src.load_config import CONFIG

BATCH_SIZE = CONFIG['train']['batch_size']
NUM_CLASSES = CONFIG['data']['num_classes']
ATOM_DIM = CONFIG['data']['atom_dim']
MLP_HIDDEN_DIM = CONFIG['model']['mlp']['hidden_dim']
BLOCK_FFN_HIDDEN_DIM = CONFIG['model']['block']['ffn']['hidden_dim']
DROPOUT_P = CONFIG['train']['dropout']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            # todo: ATOM_DIM
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, embeds):
        return self.net(embeds)


class GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=None):
        super(GRU, self).__init__()
        self.out_dim = out_dim
        dim = in_dim if out_dim is None else out_dim
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hid_dim, dim),
            nn.Dropout(p=DROPOUT_P)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim)

    def messaging(self, adjs, embeds):
        return torch.bmm(adjs, embeds)  # 批量矩阵乘法，只适用于稠密矩阵

    def forward(self, adjs, embeds):
        embeds = self.norm1(embeds + self.messaging(adjs, embeds))
        if self.out_dim is None:
            embeds = self.norm2(embeds + self.ffn(embeds))
        else:
            embeds = self.norm2(self.ffn(embeds))
        return embeds


class GRUPYG(nn.Module):
    def __init__(self):
        super(GRUPYG, self).__init__()

        self.ffn = nn.Sequential(
            # todo: ATOM_DIM
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



class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 生成Q, K, V并分头
        Q = torch.relu(self.q_norm(self.q_proj(x))) \
            .view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = torch.relu(self.k_norm(self.k_proj(x))) \
            .view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用mask
        if mask is not None:
            # 将mask转换为注意力分数对应的形状 (batch_size, 1, seq_len)
            mask = mask.unsqueeze(1)
            mask = mask.to(dtype=torch.bool)  # 确保是布尔类型
            scores = scores.masked_fill(mask, -1e9)
            scores = scores.transpose(-1, -2)
            scores = scores.masked_fill(mask, -1e9)
            scores = scores.transpose(-1, -2)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 合并多头注意力权重（平均）
        combined_weights = attention_weights.mean(dim=1)

        return combined_weights


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


class GATConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)

        # self.a = nn.Parameter(torch.Tensor(2 * out_features))
        # 分解注意力参数为两个向量（数学等效变换）
        self.a_src = nn.Parameter(torch.Tensor(out_features))
        self.a_dst = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W.weight)
        # 正确初始化1D参数向量
        # nn.init.normal_(self.a, mean=0.0, std=0.1)  # 使用正态分布初始化
        # 或者使用小范围均匀分布：
        # nn.init.uniform_(self.a, -0.01, 0.01)
        nn.init.normal_(self.a_src, mean=0.0, std=0.1)
        nn.init.normal_(self.a_dst, mean=0.0, std=0.1)

    def forward(self, adj, x):
        # x: (batch_size, num_nodes, in_features)
        # adj: (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, _ = x.size()
        # print(x.size())

        # 线性变换
        h = self.W(x)  # (b, s, out_features)

        # 生成所有节点对的特征拼接
        # h_i = h.unsqueeze(2)  # (b, s, 1, out_features)
        # h_j = h.unsqueeze(1)  # (b, 1, s, out_features)
        # concat_features = torch.cat([h_i.expand(-1, -1, num_nodes, -1), h_j.expand(-1, num_nodes, -1, -1)], dim=-1)  # (b, s, s, 2*out_features)

        # 计算注意力系数
        # e = torch.einsum('bsso,o->bss', concat_features, self.a)  # (b, s, s)
        # e = torch.matmul(concat_features, self.a)  # (b, s, s)

        # 分解式注意力计算（避免显式拼接）
        # 计算 h_i * a_src 部分 [显存：B*N*F]
        src_scores = torch.matmul(h, self.a_src)  # (B, N)

        # 计算 h_j * a_dst 部分 [显存：B*N*F]
        dst_scores = torch.matmul(h, self.a_dst)  # (B, N)

        # 广播相加 [显存：B*N*N]
        e = src_scores.unsqueeze(2) + dst_scores.unsqueeze(1)  # (B, N, N)

        e = F.leaky_relu(e, negative_slope=0.2)

        # 应用邻接矩阵掩码
        e = e.masked_fill(adj == 0, -1e9)

        # Softmax归一化
        alpha = F.softmax(e, dim=-1)  # (b, s, s)

        alpha = alpha.masked_fill(adj == 0, 0)

        # 特征聚合
        h_new = torch.bmm(alpha, h)  # (b, s, out_features)

        # 应用激活函数并返回
        return F.elu(h_new)


class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GINConv, self).__init__()
        # 初始化可学习的epsilon参数，初始值为0
        self.eps = nn.Parameter(torch.tensor([0.0]))
        # 定义MLP，包含两个线性层和ReLU激活
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.BatchNorm1d(2 * CONFIG['data']['clip_num_atom'] * out_dim),
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1, out_dim)),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, adj, x):
        # x的形状：(batch_size, num_nodes, in_dim)
        # adj的形状：(batch_size, num_nodes, num_nodes)，不包含自环

        # 聚合邻居特征，矩阵乘法
        neigh_agg = torch.bmm(adj, x)  # (b, s, in_dim)
        # 计算自身特征部分 (1 + eps) * x
        self_agg = (1 + self.eps) * x
        # 合并邻居聚合和自身特征
        aggregated = neigh_agg + self_agg
        # 应用MLP
        out = self.mlp(aggregated)  # (b, s, out_dim)
        return out


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
