import torch
from torch import nn
from Params import args


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=args.atom_dim, nhead=args.att_head,
                                                              batch_first=True)
        self.conv1d_trans = nn.Conv1d(in_channels=args.atom_dim, out_channels=args.atom_dim, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=args.atom_dim, out_features=args.mlp_hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=args.mlp_hid_dim),
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=args.mlp_hid_dim, out_features=args.n_class)
        )

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
        history_embeds = [embeds]
        for i in range(args.block_num):
            embeds = embeds.permute(0, 2, 1)
            embeds = self.conv1d_trans(embeds)
            embeds = embeds.permute(0, 2, 1)
            embeds = torch.bmm(cnn_mask, embeds)
            embeds = self.gnn_message_passing(adj, embeds)
            embeds = self.transformer_encoder(embeds, src_key_padding_mask=mask)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        embeds = torch.sum(embeds, dim=1)

        P = self.mlp(embeds)
        P = P.squeeze()

        return P, embeds


if __name__ == '__main__':
    model = MyModel()
    embeds = torch.zeros(32, 100, 75)
    adj = torch.ones(32, 100, 100)
    mask = torch.ones(32, 100) == 0
    cnn_mask = torch.ones(32, 100, 100)
    model.train()
    ans = model(embeds, adj, mask, cnn_mask)
    print(ans)
    print(ans.size())
