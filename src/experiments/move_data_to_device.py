def move_data_to_device(data, device):
    embeds, adjs, masks, cnn_masks, targets = data
    embeds = embeds.to(device)
    adjs = adjs.to(device)
    masks = masks.to(device)
    cnn_masks = cnn_masks.to(device)
    targets = targets.to(device).long()
    return embeds, adjs, masks, cnn_masks, targets