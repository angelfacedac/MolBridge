import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from src.datasets.dataloader.collate_fn import collate_fn
from src.experiments.move_data_to_device import move_data_to_device


def get_activation(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
):
    df = pd.read_csv(data_path)
    df = df[df["label"].between(begin_class, end_class)]
    sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), num_sample_per_class)))
    sampled_df = [sampled_df.iloc[i] for i in range(len(sampled_df))]
    embeds, adjs, masks, cnn_masks, labels = collate_fn(sampled_df)
    model = torch.load(model_path, map_location=device)
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
    model(embeds, adjs, masks, cnn_masks, labels)
    handle.remove()

    return activations, labels


def t_sne_2d(activations, labels, save_path):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_activations = tsne.fit_transform(activations[0].numpy())
    # 绘制t-SNE图
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(label='Labels')
    plt.title('t-SNE Visualization of Activations')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_path)


def t_sne_3d(activations, labels, save_path):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=3, random_state=42)
    reduced_activations = tsne.fit_transform(activations[0].numpy())
    # 绘制t-SNE图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_activations[:, 0], reduced_activations[:, 1], reduced_activations[:, 2], c=labels,
               cmap="viridis", alpha=0.7)
    ax.set_title('t-SNE Visualization of Activations')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.colorbar(label='Labels')
    plt.savefig(save_path)


def draw_2d(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
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
    t_sne_2d(activations, labels, save_path)


def draw_3d(
        data_path,
        model_path,
        device,
        begin_class,
        end_class,
        num_sample_per_class,
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
    t_sne_3d(activations, labels, save_path)

