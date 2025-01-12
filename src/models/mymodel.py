import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch_geometric.nn import SAGPooling, global_add_pool
from torch_scatter import scatter_add

from src.load_config import CONFIG
from src.models import myloss, mylayers

IS_RG = CONFIG['model']['is_rg']
BATCH_SIZE = CONFIG['train']['batch_size']
NUM_BLOCK = CONFIG['model']['block']['num']
DEVICE = torch.device(CONFIG['device'])


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.mlp = mylayers.MLP()

        self.blocks = nn.ModuleList(
            [mylayers.GRU() for _ in range(NUM_BLOCK)]
        )

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=CONFIG['data']['atom_dim'],
            num_heads=CONFIG['model']['attn_heads'],
            dropout=CONFIG['train']['dropout'],
            batch_first=True
        )

        if IS_RG:
            self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(CONFIG['model']['alpha'])) for _ in range(NUM_BLOCK)])


        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, embeds, adjs, masks, cnn_masks, targets):

        history_embeds = [embeds]

        for i, block in enumerate(self.blocks):
            if IS_RG:
                _, r_adjs = self.multi_head_attention(
                    embeds, embeds, embeds,
                    key_padding_mask=masks
                )
                r_adjs = torch.bmm(cnn_masks, r_adjs)
                r_adjs = torch.bmm(r_adjs, cnn_masks)
                # alpha, bata = F.softmax(torch.stack([self.alpha, 1.0 - self.alpha], dim=0), dim=0)
                # alpha, bata = 1.0, 1.0
                alpha = F.sigmoid(self.alpha[i])
                bata = 1.0 - alpha
                sum_adjs = bata * adjs + alpha * r_adjs
            else:
                r_adjs = None
                sum_adjs = adjs

            embeds = block(sum_adjs, embeds)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        embeds = torch.bmm(cnn_masks, embeds)
        embeds = torch.sum(embeds, dim=1)

        scores = self.mlp(embeds).squeeze()

        if CONFIG['model']['is_rg_loss']:
            loss = self.loss_func(scores, targets, adjs, r_adjs)
        else:
            loss = self.loss_func(scores, targets)

        return scores, loss


class MyModelPYG(nn.Module):
    def __init__(self):
        super(MyModelPYG, self).__init__()

        self.mlp = mylayers.MLP()

        self.blocks = nn.ModuleList(
            [mylayers.GRUPYG() for _ in range(NUM_BLOCK)]
        )

        self.sagpool = SAGPooling(CONFIG['data']['atom_dim'], min_score=-1)

        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, data):
        embeds = data.x
        adj_coo = data.edge_index
        batch = data.batch
        num_graphs = data.num_graphs
        targets = torch.tensor(data.y, device=embeds.device)

        history_embeds = [embeds]
        # todo:IS_RG
        for block in self.blocks:
            embeds = block(adj_coo, embeds)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        if CONFIG['model']['is_sag']:
            embeds, adj_coo, _, batch, _, _ = self.sagpool(embeds, adj_coo, batch=batch)
            embeds = global_add_pool(embeds, batch)
        else:
            embeds = scatter_add(embeds, batch, dim=0, dim_size=num_graphs)

        scores = self.mlp(embeds).squeeze()

        loss = self.loss_func(scores, targets)

        return scores, loss


if __name__ == '__main__':
    model = MyModel()
    for name, param in model.named_parameters():
        print(name, param.shape)
