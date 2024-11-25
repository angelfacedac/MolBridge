import copy

import torch
from torch import nn
from Params import args


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=args.atom_dim, nhead=args.att_head,
                                                              batch_first=True)
        # self.conv1d_trans = nn.Conv1d(in_channels=args.atom_dim, out_channels=args.atom_dim, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=args.atom_dim, out_features=args.mlp_hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=args.mlp_hid_dim),
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=args.mlp_hid_dim, out_features=args.n_class)
        )
        # self.loss = nn.MSELoss()

        # self.norm1 = nn.LayerNorm(args.atom_dim)
        # self.norm2 = nn.LayerNorm(args.atom_dim)

        self.ffn = nn.Sequential(
            nn.Linear(args.atom_dim, 2048),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(2048, args.atom_dim),
            nn.Dropout(p=args.dropout)
        )

        self.norm1s = nn.ModuleList([nn.LayerNorm(args.atom_dim) for _ in range(args.block_num)])
        self.norm2s = nn.ModuleList([nn.LayerNorm(args.atom_dim) for _ in range(args.block_num)])

        self.ffns = nn.ModuleList([copy.deepcopy(self.ffn) for _ in range(args.block_num)])


    def gnn_message_passing(self, adj, embeds):
        return torch.bmm(adj, embeds)  # 批量矩阵乘法，只适用于稠密矩阵

    def forward(self, embeds, adj, mask, cnn_mask):
        """
        :param embeds: (batch, n_atom, atom_dim)
        :param adj: (batch, n_atom, n_atom)
        :param mask: (batch, n_atom) True or False
        :param cnn_mask: (batch, n_atom, n_atom), embeds = cnn_mask * embeds, 0 or 1
        :return:
        """
        L1 = 0
        history_embeds = [embeds]
        for i in range(args.block_num):

            # 计算注意力矩阵
            output, attn_weights = self.transformer_encoder.self_attn(
                embeds, embeds, embeds, attn_mask=None, key_padding_mask=mask, need_weights=True
            )

            # 计算RGloss
            attn_weights = torch.bmm(cnn_mask, attn_weights)
            L1 += self.loss(adj[:args.clip_n_atom, :args.clip_n_atom], attn_weights[:args.clip_n_atom, :args.clip_n_atom])
            L1 += self.loss(adj[args.clip_n_atom:, args.clip_n_atom:], attn_weights[args.clip_n_atom:, args.clip_n_atom:])

            # 根据输入的邻接矩阵加上邻居节点信息 + 残差 + 层归一
            embeds = self.norm1s[i](embeds + self.gnn_message_passing(adj, embeds)) # adj + attn_weights

            # 只加下面两行，相当于transformerEncoderLayer
            # embeds = self.norm1(embeds + output)
            embeds = self.norm2s[i](embeds + self.ffns[i](embeds))

            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        embeds = torch.bmm(cnn_mask, embeds)
        embeds = torch.sum(embeds, dim=1)

        P = self.mlp(embeds)
        P = P.squeeze()

        return P, embeds, L1 * 0.5 * (1./args.block_num)


if __name__ == '__main__':
    model = MyModel()
    embeds = torch.zeros(32, 100, 75)
    adj = torch.ones(32, 100, 100)
    mask = torch.ones(32, 100) == 0
    cnn_mask = torch.ones(32, 100, 100)
    model.train()
    ans = model(embeds, adj, mask, cnn_mask)
    print(ans)
