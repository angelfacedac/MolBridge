import torch

from src.load_config import CONFIG
from src.experiments.move_data_to_device import move_data_to_device

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
