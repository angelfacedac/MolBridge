import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling

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
            [mylayers.Block() for _ in range(NUM_BLOCK)]
        )

        self.reconstructGraph = mylayers.ReconstructGraph()

        if IS_RG:
            self.alpha = nn.Parameter(torch.tensor(0.0))

        # self.sagpooling =SAGPooling()

        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, embeds, adjs, masks, cnn_masks, targets):

        history_embeds = [embeds]

        if IS_RG:
            r_adjs = self.reconstructGraph(embeds, embeds, cnn_masks)
            # alpha, bata = F.softmax(torch.stack([self.alpha, 1.0 - self.alpha], dim=0), dim=0)
            alpha, bata = 1.0, 1.0
            sum_adjs = bata * adjs + alpha * r_adjs
        else:
            r_adjs = None
            sum_adjs = adjs

        for block in self.blocks:
            embeds = block(sum_adjs, embeds)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        embeds = torch.bmm(cnn_masks, embeds)
        embeds = torch.sum(embeds, dim=1)

        scores = self.mlp(embeds).squeeze()

        if IS_RG:
            loss = self.loss_func(scores, targets, adjs, r_adjs)
        else:
            loss = self.loss_func(scores, targets)

        return scores, loss


if __name__ == '__main__':
    model = MyModel()
    for name, param in model.named_parameters():
        print(name, param.shape)
