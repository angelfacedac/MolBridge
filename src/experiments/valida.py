import torch

from src.experiments.move_data_to_device import move_data_to_device
from src.load_config import CONFIG

NUM_CLASSES = CONFIG['data']['num_classes']
DEVICE = torch.device(CONFIG['device'])


def valid(model, dataloader):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        y_true = torch.empty(0).to(DEVICE)
        y_pred = torch.empty(0).to(DEVICE)
        y_scores = torch.empty(0, NUM_CLASSES).to(DEVICE)
        for data in dataloader:
            embeds, adjs, masks, cnn_masks, targets = move_data_to_device(data, DEVICE)
            scores, loss = model(embeds, adjs, masks, cnn_masks, targets)
            sum_loss += loss.item()

            y_true = torch.cat((y_true, targets.to(DEVICE)), dim=0)
            y_scores = torch.cat((y_scores, scores), dim=0)
            y_pred = torch.cat((y_pred, torch.argmax(scores, dim=1)), dim=0)

        return sum_loss, y_true, y_pred, y_scores


def valid_pyg(model, dataloader):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        y_true = torch.empty(0).to(DEVICE)
        y_pred = torch.empty(0).to(DEVICE)
        y_scores = torch.empty(0, NUM_CLASSES).to(DEVICE)
        for data in dataloader:
            data = data.to(DEVICE)
            target = torch.tensor(data.y, device=DEVICE)
            scores, loss = model(data)
            sum_loss += loss.item()

            y_true = torch.cat((y_true, target.long()), dim=0)
            y_scores = torch.cat((y_scores, scores), dim=0)
            y_pred = torch.cat((y_pred, torch.argmax(scores, dim=1)), dim=0)

        return sum_loss, y_true, y_pred, y_scores

