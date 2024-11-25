import os.path

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import torch
from sklearn.manifold import TSNE
from torch.utils.data import Dataset

from Params import args

import matplotlib.pyplot as plt
from rdkit.Chem.rdchem import BondType
import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

    return intervals


def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']

reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]

intervals = get_intervals(reference_lists)


def get_feature_list(atom):
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list,
                             atom.GetNumRadicalElectrons())
    features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
    return features


def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals"""
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]

    # Allow 0 index to correspond to null molecule 1
    id = id + 1
    return id


def atom_to_id(atom):
    """Return a unique id corresponding to the atom type"""
    features = get_feature_list(atom)
    return features_to_id(features, intervals)


def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:

        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H?
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def smile_to_graph(smile):
    """
    :param smile:
    :return: 分子的结点特征矩阵（原子数*原子dim）， 邻接矩阵（原子数*原子数）
    """
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    return node_features, adjacency


def norm_adj(adj):
    # 计算入度
    in_degrees = torch.sum(adj, dim=0)  # 沿着第0维求和
    # D_in = torch.diag(in_degrees)  # 创建对角矩阵

    # 计算出度
    out_degrees = torch.sum(adj, dim=1)  # 沿着第1维求和
    # D_out = torch.diag(out_degrees)  # 创建对角矩阵

    # 处理零值，避免 0^(-0.5) 导致 NaN
    in_degrees = in_degrees + (in_degrees == 0).float() * 1e-10
    out_degrees = out_degrees + (out_degrees == 0).float() * 1e-10

    # 计算 -0.5 次方的矩阵
    D_in_norm = torch.diag(torch.pow(in_degrees, -0.5))
    D_out_norm = torch.diag(torch.pow(out_degrees, -0.5))

    # 左乘 D_out_neg_half 和右乘 D_in_neg_half
    result = torch.mm(D_out_norm, torch.mm(adj, D_in_norm))

    return result


class Mydata(Dataset):
    def __init__(self, root_path, kind_path):
        super(Mydata, self).__init__()
        self.root_path = root_path
        self.kind_path = kind_path
        self.path = os.path.join(root_path, kind_path)
        self.df = pd.read_csv(self.path)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)


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
        embed, adj = smile_to_graph(smile)

        # 数据对齐
        n_atom = adj.shape[0]
        new_embed = torch.zeros(args.clip_n_atom, args.atom_dim)
        new_adj = torch.zeros(args.clip_n_atom, args.clip_n_atom)
        new_cnn_mask = torch.zeros(args.clip_n_atom, args.clip_n_atom)

        pad_or_cut_size = min(n_atom, args.clip_n_atom)

        embed = torch.from_numpy(embed).float()
        new_embed[:pad_or_cut_size, :] = embed[:pad_or_cut_size, :]

        adj = torch.from_numpy(adj).float()
        new_adj[:pad_or_cut_size, :pad_or_cut_size] = adj[:pad_or_cut_size, :pad_or_cut_size]
        if args.adj_is_norm:
            new_adj = norm_adj(new_adj)

        mask = torch.tensor([False] * pad_or_cut_size + [True] * (args.clip_n_atom - pad_or_cut_size))

        cnn_mask = torch.eye(pad_or_cut_size)
        new_cnn_mask[:pad_or_cut_size, :pad_or_cut_size] = cnn_mask

        embed_dict[smile] = new_embed
        adj_dict[smile] = new_adj
        mask_dict[smile] = mask
        cnn_mask_dict[smile] = new_cnn_mask

    embeds = torch.zeros(args.batch, 2 * args.clip_n_atom, args.atom_dim)
    adjs = torch.zeros(args.batch, 2 * args.clip_n_atom, 2 * args.clip_n_atom)
    masks = torch.zeros(args.batch, 2 * args.clip_n_atom)
    cnn_masks = torch.zeros(args.batch, 2 * args.clip_n_atom, 2 * args.clip_n_atom)
    targets = torch.zeros(args.batch)

    for i, (smile1, smile2, target) in enumerate(batch):
        embed1, adj1, mask1, cnn_mask1 = embed_dict[smile1], adj_dict[smile1], mask_dict[smile1], cnn_mask_dict[smile1]
        embed2, adj2, mask2, cnn_mask2 = embed_dict[smile2], adj_dict[smile2], mask_dict[smile2], cnn_mask_dict[smile2]

        embed = torch.zeros(2 * args.clip_n_atom, args.atom_dim)
        embed[:args.clip_n_atom, :] = embed1
        embed[args.clip_n_atom:, :] = embed2

        adj = torch.zeros(2 * args.clip_n_atom, 2 * args.clip_n_atom)
        adj[:args.clip_n_atom, :args.clip_n_atom] = adj1
        adj[args.clip_n_atom:, args.clip_n_atom:] = adj2

        mask = torch.zeros(2 * args.clip_n_atom)
        mask[:args.clip_n_atom] = mask1
        mask[args.clip_n_atom:] = mask2

        cnn_mask = torch.zeros(2 * args.clip_n_atom, 2 * args.clip_n_atom)
        cnn_mask[:args.clip_n_atom, :args.clip_n_atom] = cnn_mask1
        cnn_mask[args.clip_n_atom:, args.clip_n_atom:] = cnn_mask2

        embeds[i] = embed
        adjs[i] = adj
        masks[i] = mask
        cnn_masks[i] = cnn_mask
        targets[i] = target

    embeds = embeds.cuda(args.gpu)
    adjs = adjs.cuda(args.gpu)
    masks = masks.cuda(args.gpu)
    cnn_masks = cnn_masks.cuda(args.gpu)
    targets = targets.cuda(args.gpu).long()

    return embeds, adjs, masks, cnn_masks, targets


def T_SNE(embeds, labels, path=None):
    tsne = TSNE(n_components=2)
    embeds = tsne.fit_transform(embeds)

    # 可视化
    # plt.figure(figsize=(8, 6))
    plt.scatter(embeds[:, 0], embeds[:, 1], c=labels, cmap=plt.cm.get_cmap("tab20", len(set(labels))))
    plt.colorbar(ticks=range(len(set(labels))))
    plt.title("t-SNE visualization of drug pair embed")
    plt.show()
    if path:
        plt.savefig(path)


if __name__ == '__main__':
    smile = 'OC(CN1C=NC=N1)(CN1C=NC=N1)C1=C(F)C=C(F)C=C1'
    smile = 'C[C@@H]1CCN([C@H](C1)C(O)=O)C(=O)[C@H](CCCN=C(N)N)NS(=O)(=O)C1=CC=CC2=C1NC[C@H](C)C2'
    node_embeding, adj = smile_to_graph(smile)
    print(node_embeding)
    print(node_embeding.shape)
    print(adj)
    print(adj.shape[0])
