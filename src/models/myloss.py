import torch
import torch.nn.functional as F
from torch import nn

from src.load_config import CONFIG

LOSS_FN = CONFIG['train']['loss_fn']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterion = getattr(nn, LOSS_FN)()
        if CONFIG['model']['is_rg_loss']:
            self.criterion_adjs = nn.BCEWithLogitsLoss()
            self.alpha = nn.Parameter(torch.tensor(CONFIG['model']['alpha']))

    def forward(self, scores, targets, adjs=None, r_adjs=None):
        loss1 = self.criterion(scores, targets.long())

        if CONFIG['model']['is_rg_loss']:
            loss2 = self.criterion_adjs(
                r_adjs[:, :CLIP_NUM_ATOM, :CLIP_NUM_ATOM],
                adjs[:, :CLIP_NUM_ATOM, :CLIP_NUM_ATOM]
            ) + self.criterion_adjs(
                r_adjs[:, CLIP_NUM_ATOM:, CLIP_NUM_ATOM:],
                adjs[:, CLIP_NUM_ATOM:, CLIP_NUM_ATOM:]
            )
            # alpha, bata = F.softmax(torch.stack([self.alpha, 1.0 - self.alpha], dim=0), dim=0)
            # alpha, bata = 1.0, 1.0
            alpha = F.sigmoid(self.alpha)
            bata = 1.0 - alpha
            return bata * loss1 + alpha * loss2
        return loss1




