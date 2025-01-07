import torch
from torch.utils.tensorboard import SummaryWriter

from src.experiments.move_data_to_device import move_data_to_device
from src.models.mymodel import MyModel
from src.load_config import CONFIG

ATOM_DIM = CONFIG['data']['atom_dim']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']
DEVICE = torch.device(CONFIG['device'])


def add_graph(tensorboard_path):
    model = MyModel()
    model.to(DEVICE)
    writer = SummaryWriter(tensorboard_path)
    embeds = torch.zeros(2, 2 * CLIP_NUM_ATOM, ATOM_DIM)
    adj = torch.zeros(2, 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
    masks = torch.zeros(2, 2 * CLIP_NUM_ATOM)
    cnn_masks = torch.zeros(2, 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
    targets = torch.zeros(2)
    embeds, adj, masks, cnn_masks, targets = move_data_to_device([embeds, adj, masks, cnn_masks, targets], DEVICE)
    writer.add_graph(model, [embeds, adj, masks, cnn_masks, targets])
    writer.close()

