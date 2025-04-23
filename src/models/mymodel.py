import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch_geometric.nn import SAGPooling, global_add_pool, GATConv
from torch_scatter import scatter_add

from src.load_config import CONFIG
from src.models import myloss, mylayers
from src.models.GCN import GraphConvolution

IS_RG = CONFIG['model']['is_rg']
BATCH_SIZE = CONFIG['train']['batch_size']
NUM_BLOCK = CONFIG['model']['block']['num']
DEVICE = torch.device(CONFIG['device'])
NUM_CLASSES = CONFIG['data']['num_classes']
ATOM_DIM = CONFIG['data']['atom_dim']
MLP_HIDDEN_DIM = CONFIG['model']['mlp']['hidden_dim']
BLOCK_FFN_HIDDEN_DIM = CONFIG['model']['block']['ffn']['hidden_dim']
DROPOUT_P = CONFIG['train']['dropout']
CLIP_NUM_ATOM = CONFIG['data']['clip_num_atom']
ATOM_HID_DIM = CONFIG['data']['atom_hid_dim']


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        if CONFIG['model']['is_cat_after_readout']:
            self.mlp = mylayers.MLP(ATOM_HID_DIM * (NUM_BLOCK+1), MLP_HIDDEN_DIM, NUM_CLASSES)
        else:
            self.mlp = mylayers.MLP(ATOM_HID_DIM, MLP_HIDDEN_DIM, NUM_CLASSES)

        if CONFIG['model']['is_gcn']:
            self.blocks = nn.ModuleList(
                [GraphConvolution(CONFIG['data']['atom_hid_dim'], CONFIG['data']['atom_hid_dim']) for _ in range(NUM_BLOCK)]
            )
        elif CONFIG['model']['is_gat']:
            self.blocks = nn.ModuleList(
                [mylayers.GATConv(CONFIG['data']['atom_hid_dim'], CONFIG['data']['atom_hid_dim']) for _ in range(NUM_BLOCK)]
            )
        elif CONFIG['model']['is_gin']:
            self.blocks = nn.ModuleList(
                [mylayers.GINConv(CONFIG['data']['atom_hid_dim'], CONFIG['data']['atom_hid_dim']) for _ in range(NUM_BLOCK)]
            )
        else:
            self.blocks = nn.ModuleList(
                [mylayers.GRU(ATOM_HID_DIM, BLOCK_FFN_HIDDEN_DIM) for _ in range(NUM_BLOCK)]
            )

        if CONFIG['model']['is_init_gru']:
            self.init = mylayers.GRU(ATOM_DIM, BLOCK_FFN_HIDDEN_DIM, ATOM_HID_DIM)
        else:
            self.init = nn.Sequential(
                nn.Linear(ATOM_DIM, ATOM_HID_DIM),
                nn.LayerNorm(ATOM_HID_DIM),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_P)
            )

        if IS_RG:
            self.multi_head_attention = nn.MultiheadAttention(
                    embed_dim=ATOM_HID_DIM,
                    num_heads=CONFIG['model']['attn_heads'],  # 4 is before the time
                    dropout=CONFIG['train']['dropout'],
                    batch_first=True
                )
            # self.multi_head_attention = mylayers.SelfAttention(ATOM_HID_DIM, CONFIG['model']['attn_heads'])

            if CONFIG['model']['is_alpha_learn']:
                self.alpha = nn.Parameter(torch.tensor(CONFIG['model']['alpha']))
            else:
                self.alpha = CONFIG['model']['alpha']

        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def cal(self, embeds, adjs, masks):

        if CONFIG['model']['is_init_gru']:
            embeds = self.init(adjs, embeds)
        else:
            embeds = self.init(embeds)

        history_embeds = [embeds]

        if IS_RG:
            _, r_adjs = self.multi_head_attention(
                embeds.detach(), embeds.detach(), embeds.detach(),
                key_padding_mask=masks
            )
            # r_adjs = torch.bmm(cnn_masks, r_adjs)
            # r_adjs = torch.bmm(r_adjs, cnn_masks)
            r_adjs = r_adjs.masked_fill(masks.unsqueeze(2).to(dtype=torch.bool), 0)
            r_adjs = r_adjs.masked_fill(masks.unsqueeze(1).to(dtype=torch.bool), 0)
            # r_adjs = self.multi_head_attention(embeds, masks)
            # r_adjs = r_adjs.masked_fill(adjs.to(dtype=torch.bool), 0)
            if CONFIG['model']['is_a_']:
                if CONFIG['model']['is_alpha_learn']:
                    alpha = F.sigmoid(self.alpha)
                else:
                    alpha = self.alpha
                bata = 1.0 - alpha
                sum_adjs = bata * adjs + alpha * r_adjs
            else:
                sum_adjs = r_adjs
        else:
            sum_adjs = adjs

        for i, block in enumerate(self.blocks):
            embeds = block(sum_adjs, embeds)
            history_embeds.append(embeds)

        if CONFIG['model']['is_cat_after_readout']:
            embeds = torch.cat(history_embeds, dim=-1)
        else:
            embeds = sum(history_embeds)
        del history_embeds

        # embeds = torch.bmm(cnn_masks, embeds)
        embeds = embeds.masked_fill(masks.unsqueeze(2).to(dtype=torch.bool), 0)
        embeds = torch.sum(embeds, dim=1)
        return embeds

    def cal1(self, embeds, adjs, masks):

        embeds = self.init(embeds)
        em1 = embeds
        em2 = embeds
        history_em1 = [embeds]
        history_em2 = [embeds]
        _, r_adjs = self.multi_head_attention(
            embeds.detach(), embeds.detach(), embeds.detach(),
            key_padding_mask=masks
        )
        r_adjs = r_adjs.masked_fill(masks.unsqueeze(2).to(dtype=torch.bool), 0)
        r_adjs = r_adjs.masked_fill(masks.unsqueeze(1).to(dtype=torch.bool), 0)

        for i, block in enumerate(self.blocks):
            em1 = block(adjs, em1)
            history_em1.append(em1)

        for i, block in enumerate(self.blocks):
            em2 = block(r_adjs, em2)
            history_em2.append(em2)

        em1 = sum(history_em1)
        em2 = sum(history_em2)

        embeds = torch.cat([em1, em2], dim=-1)
        embeds = embeds.masked_fill(masks.unsqueeze(2).to(dtype=torch.bool), 0)
        embeds = torch.sum(embeds, dim=1)
        return embeds

    def forward(self, embeds, adjs, masks, cnn_masks, targets):

        if CONFIG['model']['is_joint']:
            embeds = self.cal(embeds, adjs, masks)

        else:
            clip_size = CONFIG['data']['clip_num_atom']
            embeds1 = self.cal(embeds[:, :clip_size, :], adjs[:, :clip_size, :clip_size], masks[:, :clip_size])
            embeds2 = self.cal(embeds[:, clip_size:, :], adjs[:, clip_size:, clip_size:], masks[:, clip_size:])
            embeds = embeds1 + embeds2

        scores = self.mlp(embeds)

        loss = self.loss_func(scores, targets)

        return scores, loss
        # return scores, r_adjs


class MyModelPYG(nn.Module):
    def __init__(self):
        super(MyModelPYG, self).__init__()

        self.mlp = mylayers.MLP()

        self.blocks = nn.ModuleList(
            [mylayers.GRUPYG() for _ in range(NUM_BLOCK)]
        )

        if IS_RG:
            self.gat_o = GATConv(CONFIG['data']['atom_dim'], 32, 4)
            self.gat_i = GATConv(128, 32, 4)

            self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(CONFIG['model']['alpha'])) for _ in range(NUM_BLOCK)])

        if CONFIG['model']['is_sag']:
            # todo:CONFIG['data']['atom_dim']
            self.sagpool = SAGPooling(CONFIG['data']['atom_dim'], min_score=-1)

        self.loss_func = myloss.MyLoss()

        # self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # init.kaiming_uniform_(m.weight, nonlinearity='relu')
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def create_complete_graph_edge_index(self, num_nodes):
        # 创建节点索引的所有组合，生成一个完全图的边索引
        node_indices = torch.arange(num_nodes)
        edge_index = torch.combinations(node_indices, r=2).T  # 生成所有节点对
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        return edge_index

    def batch_create_complete_graph_edge_index(self, batch, num_graphs):
        # 为每个图创建完全图的边索引
        new_edge_index_list = []
        offset = 0  # 用于节点索引的偏移量
        for i in range(num_graphs):
            # 获取当前图的节点数
            num_nodes = (batch == i).sum().item()

            # 为当前图创建完全图的 edge_index
            edge_index = self.create_complete_graph_edge_index(num_nodes)

            # 将当前图的节点索引偏移量加到 edge_index 上
            edge_index += offset

            # 将创建的 edge_index 添加到列表中
            new_edge_index_list.append(edge_index)

            # 更新节点偏移量
            offset += num_nodes

        # 将所有生成的完全图的 edge_index 合并为一个张量
        new_edge_index = torch.cat(new_edge_index_list, dim=-1)
        return new_edge_index

    def forward(self, data):
        embeds = data.x
        adj_coo = data.edge_index
        batch = data.batch
        num_graphs = data.num_graphs
        targets = torch.tensor(data.y, device=embeds.device)

        embeds = self.gat_o(embeds, adj_coo)

        history_embeds = [embeds]

        if IS_RG:
            comp_adj_coo = self.batch_create_complete_graph_edge_index(batch, num_graphs)
            comp_adj_coo = comp_adj_coo.to(embeds.device)


        for i, block in enumerate(self.blocks):
            if IS_RG:
                alpha = F.sigmoid(self.alpha[i])
                embeds = (1 - alpha) * embeds + alpha * self.gat_i(embeds, comp_adj_coo)

            embeds = block(adj_coo, embeds)
            history_embeds.append(embeds)

        embeds = sum(history_embeds)
        del history_embeds

        if CONFIG['model']['is_sag']:
            embeds, adj_coo, _, batch, _, _ = self.sagpool(embeds, adj_coo, batch=batch)
            embeds = global_add_pool(embeds, batch)
        else:
            embeds = scatter_add(embeds, batch, dim=0, dim_size=num_graphs)

        scores = self.mlp(embeds).squeeze()

        loss = self.loss_func(scores, targets)

        return scores, loss


if __name__ == '__main__':
    model = MyModel()
    for name, param in model.named_parameters():
        print(name, param.shape)
