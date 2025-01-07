import torch
from torch import nn
import torch.nn.functional as F

from src.load_config import CONFIG

IS_RG = CONFIG['model']['is_rg']
LOSS_FN = CONFIG['train']['loss_fn']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterion = getattr(nn, LOSS_FN)()
        if IS_RG:
            self.criterion_adjs = nn.BCEWithLogitsLoss()
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, scores, targets, adjs=None, r_adjs=None):
        loss1 = self.criterion(scores, targets.long())

        if IS_RG:
            loss2 = self.criterion_adjs(
                r_adjs[:, :CLIP_NUM_ATOM, :CLIP_NUM_ATOM],
                adjs[:, :CLIP_NUM_ATOM, :CLIP_NUM_ATOM]
            ) + self.criterion_adjs(
                r_adjs[:, CLIP_NUM_ATOM:, CLIP_NUM_ATOM:],
                adjs[:, CLIP_NUM_ATOM:, CLIP_NUM_ATOM:]
            )

            # alpha, bata = F.softmax(torch.stack([self.alpha, 1.0 - self.alpha], dim=0), dim=0)
            alpha, bata = 1.0, 1.0
            return bata * loss1 + alpha * loss2
        return loss1

