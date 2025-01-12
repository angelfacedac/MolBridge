import torch

from src.experiments.move_data_to_device import move_data_to_device
from src.load_config import CONFIG

DEVICE = torch.device(CONFIG['device'])

def train(model, dataloader, opt):
    model.train()
    sum_loss = 0
    for data in dataloader:
        embeds, adjs, masks, cnn_masks, targets = move_data_to_device(data, DEVICE)
        scores, loss = model(embeds, adjs, masks, cnn_masks, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_loss += loss.item()

    return sum_loss


def train_pyg(model, dataloader, opt):
    model.train()
    sum_loss = 0
    for data in dataloader:
        data = data.to(DEVICE)
        scores, loss = model(data)
        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_loss += loss.item()

    return sum_loss

