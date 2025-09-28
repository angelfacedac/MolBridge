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

    def forward(self, scores, targets):
        if CONFIG['data']['is_binary']:
            # 二分类：使用BCEWithLogitsLoss，targets需要转换为float
            loss1 = self.criterion(scores.squeeze(), targets.float())
        else:
            # 多分类：使用CrossEntropyLoss，targets转换为long
            loss1 = self.criterion(scores, targets.long())
        return loss1




