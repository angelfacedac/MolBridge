import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch

from src.experiments.move_data_to_device import move_data_to_device
from src.load_config import CONFIG
from src.models.mymodel import MyModel, MyModelPYG

ATOM_DIM = CONFIG['data']['atom_dim']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']
DEVICE = torch.device(CONFIG['device'])


def add_graph(tensorboard_path):
    writer = SummaryWriter(tensorboard_path)

    # todo:
    if CONFIG['is_pyg']:
        model = MyModelPYG()
        x1 = torch.zeros(2, ATOM_DIM)
        edge_index1 = torch.tensor([[0, 1],
                                    [1, 1]], dtype=torch.long)
        data1 = Data(x=x1, edge_index=edge_index1, y=torch.tensor([0, 1]))
        x2 = torch.zeros(2, ATOM_DIM)
        edge_index2 = torch.tensor([[0, 1],
                                    [1, 0]], dtype=torch.long)
        data2 = Data(x=x2, edge_index=edge_index2, y=torch.tensor([0, 1]))
        data = Batch.from_data_list([data1, data2])
        data = data.to(DEVICE)
    else:
        model = MyModel()
        embeds = torch.zeros(2, 2 * CLIP_NUM_ATOM, ATOM_DIM)
        adj = torch.zeros(2, 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
        masks = torch.zeros(2, 2 * CLIP_NUM_ATOM)
        cnn_masks = torch.zeros(2, 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
        targets = torch.zeros(2)
        embeds, adj, masks, cnn_masks, targets = move_data_to_device([embeds, adj, masks, cnn_masks, targets], DEVICE)
        data = [embeds, adj, masks, cnn_masks, targets]
    model.to(DEVICE)
    # print(data)
    writer.add_graph(model, data)
    writer.close()

