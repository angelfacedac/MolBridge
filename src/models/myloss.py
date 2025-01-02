from torch import nn

from src.load_config import CONFIG

LOSS_FN = CONFIG['train']['loss_fn']


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterion = getattr(nn, LOSS_FN)()
        nn.CrossEntropyLoss

    def forward(self, scores, targets):
        return self.criterion(scores, targets.long())

