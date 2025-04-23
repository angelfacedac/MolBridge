import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE

from src.datasets.dataloader.collate_fn import collate_fn
from src.experiments.move_data_to_device import move_data_to_device
from src.models import mymodel

plt.rcParams['font.family'] = 'Times New Roman'

def get_activation(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
):
    # df = pd.read_csv(data_path)
    # df = df[df["label"].between(begin_class, end_class)]
    # sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), num_sample_per_class)))

    sampled_df = pd.read_csv("sample-tsne.csv")
    sampled_df = sampled_df.loc[:, ['smile_1', 'smile_2', 'label']]

    sampled_df = [sampled_df.iloc[i] for i in range(len(sampled_df))]
    embeds, adjs, masks, cnn_masks, labels = collate_fn(sampled_df)
    model = mymodel.MyModel()
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True
        )
    )
    model.to(device)
    print(model)

    activations = []

    def forward_hook(module, input, output):
        activations.append(input[0].detach())

    handle = model.mlp.register_forward_hook(forward_hook)
    embeds, adjs, masks, cnn_masks, labels = move_data_to_device(
        (embeds, adjs, masks, cnn_masks, labels),
        device
    )
    init_embeds = torch.sum(embeds, dim=1)
    activations.append(init_embeds)
    model(embeds, adjs, masks, cnn_masks, labels)
    handle.remove()

    return activations, labels


def t_sne_2d(activations, labels, colors, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(10, 20))
    if type(colors) == list:
        colors = ListedColormap(colors)
    for k, title in zip([0, 1], ['Before GRN-DDI', 'After GRN-DDI']):
        embeds = activations[k]
        reduced_activations = tsne.fit_transform(embeds.cpu().numpy())
        ax = plt.subplot(2, 1, k + 1)
        ax.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=labels, cmap=colors, alpha=0.7)
        ax.tick_params(axis='x', labelsize=35)
        ax.tick_params(axis='y', labelsize=35)
        # ax.axis('off')  # 去掉坐标轴
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_title(title, fontsize=30, fontweight='bold')

    # plt.colorbar(label='Labels')
    # plt.title('t-SNE Visualization of Activations')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path)


def t_sne_2d_single(activations, labels, colors, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(10, 10))
    if type(colors) == list:
        colors = ListedColormap(colors)

    embeds = activations[1]
    embeds = embeds.cpu().numpy()
    reduced_activations = tsne.fit_transform(embeds)
    ax = plt.subplot(111)
    ax.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=labels, cmap=colors, alpha=0.7)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    # ax.axis('off')  # 去掉坐标轴
    # ax.set_xticks([])  # 去掉x轴刻度
    # ax.set_yticks([])  # 去掉y轴刻度
    # ax.set_title(title, fontsize=30, fontweight='bold')

    # plt.colorbar(label='Labels')
    # plt.title('t-SNE Visualization of Activations')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.tight_layout()  # 自动调整子图参数
    plt.show()
    # plt.savefig(save_path)


def t_sne_3d(activations, labels, colors, save_path):
    tsne = TSNE(n_components=3, random_state=42)
    plt.figure(figsize=(10, 20))  # 设置图形大小
    if type(colors) == list:
        colors = ListedColormap(colors)
    for k, title in zip([0, 1], ['Before GRN-DDI', 'After GRN-DDI']):
        embeds = activations[k]
        reduced_activations = tsne.fit_transform(embeds.cpu().numpy())
        ax = plt.subplot(2, 1, k + 1, projection='3d')  # 创建3D子图
        ax.scatter3D(reduced_activations[:, 0], reduced_activations[:, 1], reduced_activations[:, 2], c=labels,
                   cmap=colors, alpha=0.7)
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_zticks([])  # 去掉z轴刻度
        ax.view_init(elev=30, azim=-60)  # 调整视角
        ax.set_title(title, fontsize=30, fontweight='bold')


    # plt.colorbar(label='Labels', ax=ax, location='right')  # 添加颜色条
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path)


def draw_2d(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
        colors,
        save_path
):
    activations, labels = get_activation(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
    )
    labels = labels.cpu().numpy()
    t_sne_2d(activations, labels, colors, save_path)

def draw_2d_single(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
        colors,
        save_path
):
    activations, labels = get_activation(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
    )
    labels = labels.cpu().numpy()
    t_sne_2d_single(activations, labels, colors, save_path)


def draw_3d(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
        colors,
        save_path
):
    activations, labels = get_activation(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
    )
    labels = labels.cpu().numpy()
    t_sne_3d(activations, labels, colors, save_path)

