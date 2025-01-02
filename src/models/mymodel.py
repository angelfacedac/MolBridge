import torch
from torch import nn
from torch.nn import init
from torch_geometric.nn import SAGPooling

from src.load_config import CONFIG
from src.models import myloss, mylayers

NUM_BLOCK = CONFIG['model']['block']['num']
DEVICE = torch.device(CONFIG['device'])


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.mlp = mylayers.MLP()

        self.blocks = nn.ModuleList(
            [mylayers.Block() for _ in range(NUM_BLOCK)]
        )

        # self.sagpooling =SAGPooling()

        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, embeds, adjs, masks, cnn_masks, targets):
        embeds = embeds.to(DEVICE)
        adjs = adjs.to(DEVICE)
        masks = masks.to(DEVICE)
        cnn_masks = cnn_masks.to(DEVICE)
        targets = targets.to(DEVICE).long()

        history_embeds = [embeds]

        for block in self.blocks:
            embeds = block(adjs, embeds)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        embeds = torch.bmm(cnn_masks, embeds)
        embeds = torch.sum(embeds, dim=1)

        scores = self.mlp(embeds).squeeze()

        loss = self.loss_func(scores, targets)

        return scores, loss


if __name__ == '__main__':
    model = MyModel()
    for name, param in model.named_parameters():
        print(name, param.shape)
