import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from src.datasets.dataloader.feature_encoding import smile_to_graph
from src.datasets.dataloader.utils import norm_adj
from src.load_config import CONFIG

ATOM_DIM = CONFIG['data']['atom_dim']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']
IS_NORM_ADJ = CONFIG['model']['is_norm_adj']

all_smile = set()
embed_dict = dict()
adj_dict = dict()
mask_dict = dict()
cnn_mask_dict = dict()


def collate_fn(batch):
    global embed_dict, adj_dict, mask_dict, cnn_mask_dict, all_smile
    """embeds, adjs, masks, cnn_masks, targets = data"""

    for smile1, smile2, _ in batch:
        all_smile.add(smile1)
        all_smile.add(smile2)

    for smile in all_smile:
        if smile in embed_dict: continue
        embed, adj = smile_to_graph(smile)

        # 数据对齐
        n_atom = adj.shape[0]
        new_embed = torch.zeros(CLIP_NUM_ATOM, ATOM_DIM)
        new_adj = torch.zeros(CLIP_NUM_ATOM, CLIP_NUM_ATOM)
        new_cnn_mask = torch.zeros(CLIP_NUM_ATOM, CLIP_NUM_ATOM)

        pad_or_cut_size = min(n_atom, CLIP_NUM_ATOM)

        embed = torch.from_numpy(embed).float()
        new_embed[:pad_or_cut_size, :] = embed[:pad_or_cut_size, :]

        adj = torch.from_numpy(adj).float()

        if CONFIG['model']['is_eye']:
            identity_matrix = torch.eye(n_atom, dtype=torch.float32)
            adj = adj + identity_matrix

        new_adj[:pad_or_cut_size, :pad_or_cut_size] = adj[:pad_or_cut_size, :pad_or_cut_size]
        if IS_NORM_ADJ:
            new_adj = norm_adj(new_adj)

        mask = torch.tensor([False] * pad_or_cut_size + [True] * (CLIP_NUM_ATOM - pad_or_cut_size))

        cnn_mask = torch.eye(pad_or_cut_size)
        new_cnn_mask[:pad_or_cut_size, :pad_or_cut_size] = cnn_mask

        embed_dict[smile] = new_embed
        adj_dict[smile] = new_adj
        mask_dict[smile] = mask
        cnn_mask_dict[smile] = new_cnn_mask

    embeds = torch.zeros(len(batch), 2 * CLIP_NUM_ATOM, ATOM_DIM)
    adjs = torch.zeros(len(batch), 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
    masks = torch.zeros(len(batch), 2 * CLIP_NUM_ATOM)
    cnn_masks = torch.zeros(len(batch), 2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
    targets = torch.zeros(len(batch))

    for i, (smile1, smile2, target) in enumerate(batch):
        embed1, adj1, mask1, cnn_mask1 = embed_dict[smile1], adj_dict[smile1], mask_dict[smile1], cnn_mask_dict[smile1]
        embed2, adj2, mask2, cnn_mask2 = embed_dict[smile2], adj_dict[smile2], mask_dict[smile2], cnn_mask_dict[smile2]

        embed = torch.zeros(2 * CLIP_NUM_ATOM, ATOM_DIM)
        embed[:CLIP_NUM_ATOM, :] = embed1
        embed[CLIP_NUM_ATOM:, :] = embed2

        adj = torch.zeros(2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
        adj[:CLIP_NUM_ATOM, :CLIP_NUM_ATOM] = adj1
        adj[CLIP_NUM_ATOM:, CLIP_NUM_ATOM:] = adj2

        mask = torch.zeros(2 * CLIP_NUM_ATOM)
        mask[:CLIP_NUM_ATOM] = mask1
        mask[CLIP_NUM_ATOM:] = mask2

        cnn_mask = torch.zeros(2 * CLIP_NUM_ATOM, 2 * CLIP_NUM_ATOM)
        cnn_mask[:CLIP_NUM_ATOM, :CLIP_NUM_ATOM] = cnn_mask1
        cnn_mask[CLIP_NUM_ATOM:, CLIP_NUM_ATOM:] = cnn_mask2

        embeds[i] = embed
        adjs[i] = adj
        masks[i] = mask
        cnn_masks[i] = cnn_mask
        targets[i] = target

    return embeds, adjs, masks, cnn_masks, targets


def collate_fn_pyg(batch):
    global embed_dict, adj_dict, mask_dict, cnn_mask_dict, all_smile

    for smile1, smile2, _ in batch:
        all_smile.add(smile1)
        all_smile.add(smile2)


    for smile in all_smile:
        if smile in embed_dict: continue

        embed, adj = smile_to_graph(smile)

        # 将邻接矩阵转换为PyTorch张量
        adj = torch.tensor(adj, dtype=torch.float)

        if CONFIG['model']['is_eye']:
            identity_matrix = torch.eye(adj.shape[0], dtype=torch.float32)
            adj = adj + identity_matrix

        # 将密集的邻接矩阵转换为COO格式
        edge_index = dense_to_sparse(adj)[0]
        embed = torch.from_numpy(embed).float()

        embed_dict[smile] = embed
        adj_dict[smile] = edge_index

    batch_samples = []

    for i, (smile1, smile2, target) in enumerate(batch):
        embed1, adj1_coo = embed_dict[smile1], adj_dict[smile1]
        embed2, adj2_coo = embed_dict[smile2], adj_dict[smile2]

        embed = torch.cat([embed1, embed2], dim=0)
        adj_coo = torch.cat([adj1_coo, adj2_coo + embed1.size(0)], dim=1)

        sample = Data(x=embed, edge_index=adj_coo, y=target)

        batch_samples.append(sample)

    batch_samples = Batch.from_data_list(batch_samples)

    return batch_samples
