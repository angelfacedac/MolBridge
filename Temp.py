import os

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mymodel
from Params import args
from main import init_data, test, valida
from mymodel import MyModel
from utils import Mydata, collate_fn, T_SNE


def deal_data_for_Deng_and_Ryu():
    drug2smile = dict()
    df_d2s = pd.read_csv("Data/Deng's dataset/drug_smiles.csv")
    for i in range(len(df_d2s)):
        drug, smile = df_d2s.iloc[i]
        drug2smile[drug] = smile

    def deal(k, kind):
        df = pd.read_csv(f"Data/Ryu's dataset/{k}/ddi_{kind}1.csv")

        df[["type", "d2"]] = df[["d2", "type"]]
        df.columns = ["smile_1", "smile_2", "label"]

        for i in range(len(df)):
            drug1, drug2, label = df.iloc[i]
            smile1 = drug2smile[drug1]
            smile2 = drug2smile[drug2]
            df.iloc[i] = smile1, smile2, label

        df.to_csv(f"Data/Ryu's dataset/{k}/{kind}.csv", index=False)

    for k in range(5):
        for kind in ["test", "training", "validation"]:
            deal(k, kind)


def remove_checkpoint(date, dataset):
    for t in date:
        for i in [f"{dataset}0", f"{dataset}1", f"{dataset}2", f"{dataset}3", f"{dataset}4"]:
            root_path = f"/home/linx/Dac/TransGNN-DDI/Out/{t}/{i}/checkpoint"
            for i in range(1, 501):
                if i % 50 == 0: continue
                path = f"{root_path}/{i:03d}.pth"
                os.remove(path)
                print(f"remove {path} success!")


def add_val():
    for path in [
        "/home/linx/Dac/TransGNN-DDI/Out/2024-08-16_03-42-58_D0",
        "/home/linx/Dac/TransGNN-DDI/Out/2024-08-16_09-25-52_D1",
        "/home/linx/Dac/TransGNN-DDI/Out/2024-08-16_17-29-41_D2",
        "/home/linx/Dac/TransGNN-DDI/Out/2024-08-16_21-43-00_D3",
        "/home/linx/Dac/TransGNN-DDI/Out/2024-08-17_01-38-37_D4"
    ]:
        for epoch in range(1, 501):
            checkpoint_path = os.path.join(path, "checkpoint")
            checkpoint = torch.load(os.path.join(checkpoint_path, f"{epoch:03d}.pth"))
            model = MyModel()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(args.device)

            data = Mydata(root_path=os.path.join(args.root_path, path[-1]), kind_path=args.val_path)
            dataloader = DataLoader(data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

            writer = SummaryWriter(os.path.join(path, "tensorboard"))

            loss_fn = getattr(nn, args.loss_fn)()

            valida(model, dataloader, loss_fn, epoch, writer)


def do_T_SNE():
    model = MyModel()
    checkpoint_path = "482.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    df = pd.read_csv("Data/Deng's dataset/3/test.csv")
    df = df[df["label"].between(0, 4)]
    sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 200)))
    sampled_df = [sampled_df.iloc[i] for i in range(len(sampled_df))]
    print(len(sampled_df))

    embeds, adjs, masks, cnn_masks, labels = collate_fn(sampled_df)

    print(embeds.requires_grad)

    labels = labels.cpu().numpy()

    pre_embeds = torch.sum(embeds, dim=1)
    print(pre_embeds.requires_grad)
    pre_embeds = pre_embeds.cpu().numpy()

    _, out_embeds = model(embeds, adjs, masks, cnn_masks)
    print(out_embeds.requires_grad)
    out_embeds = out_embeds.cpu().detach().numpy()

    T_SNE(pre_embeds, labels)
    T_SNE(out_embeds, labels)


def Vis_Cal_Graph():
    model = MyModel()
    model.eval()
    embed = torch.zeros(1, args.clip_n_atom, args.atom_dim)
    adj = torch.zeros(1, args.clip_n_atom, args.clip_n_atom)
    mask = torch.zeros(1, args.clip_n_atom)
    cnn_mask = torch.zeros(1, args.clip_n_atom, args.clip_n_atom)

    writer = SummaryWriter("Out/Graph")
    writer.add_graph(model, [embed, adj, mask, cnn_mask])
    writer.close()

    output = model(embed, adj, mask, cnn_mask)

    print(output)

remove_checkpoint(
    ["2024-10-21_02-12-51",
     "2024-10-21_02-09-06",
     "2024-10-20_03-55-24",
     "2024-10-10_14-26-00",
     ],
    "D"
)
