from src.load_config import CONFIG


def train(model, dataloader, opt):
    model.train()
    sum_loss = 0
    for data in dataloader:
        embeds, adjs, masks, cnn_masks, targets = data
        scores, loss = model(embeds, adjs, masks, cnn_masks, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_loss += loss.item()

    return sum_loss
