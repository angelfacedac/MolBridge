import pandas as pd
import torch
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from mymodel import MyModel
from src.utils import collate_fn


def plot_embedding_2d(embedding_2d, labels,
                      title='2D t-SNE Visualization',
                      save_path='Out/TSNE-2D.png'):
    """
    :param embedding_2d: (N, num_features)
    :param labels: (N,)
    :param title:
    :param save_path:
    :return:
    """
    plt.figure()
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='viridis'
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(save_path)
    print("TSNE-2D.png has saved")


def plot_embedding_3d(embedding_3d, labels,
                      title='3D t-SNE Visualization',
                      save_path='Out/TSNE-3D.png'):
    """
    :param embedding_3d: (N, num_features)
    :param labels: (N,)
    :param title:
    :param save_path:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], c=labels, cmap='viridis'
    )

    plt.colorbar(scatter)
    ax.set_title(title)
    plt.savefig(save_path)
    print("TSNE-3D.png has saved")


def draw_umap(*input_tensor, labels, model, target_layer):

    embedding = None
    def forward_hook(module, input, output):
        nonlocal embedding
        embedding = input[0].detach()

    hander = target_layer.register_forward_hook(forward_hook)

    output = model(*input_tensor)
    hander.remove()

    print("embedding shape:", embedding.shape)
    num_samples = embedding.shape[0]
    embedding = embedding.reshape(num_samples, -1)

    embedding_2d = umap.UMAP(n_components=2)\
        .fit_transform(embedding)

    plot_embedding_2d(embedding_2d, labels,
                      title='2D UMAP Visualization',
                      save_path='Out/UMAP-2D.png')

    embedding_3d = umap.UMAP(n_components=3)\
        .fit_transform(embedding)

    plot_embedding_3d(embedding_3d, labels,
                      title='3D UMAP Visualization',
                      save_path='Out/UMAP-3D.png')



def draw_tsne(*input_tensor, labels, model, target_layer):
    embeddings = None
    def forward_hook(module, input, output):
        nonlocal embeddings
        embeddings = input[0].detach()

    hander = target_layer.register_forward_hook(forward_hook)

    output = model(*input_tensor)
    hander.remove()

    print("embeddings shape:", embeddings.shape)
    num_samples = embeddings.shape[0]
    embeddings = embeddings.reshape(num_samples, -1)

    embeddings_2d = TSNE(n_components=2)\
        .fit_transform(embeddings)

    plot_embedding_2d(embeddings_2d, labels)

    embeddings_3d = TSNE(n_components=3)\
        .fit_transform(embeddings)

    plot_embedding_3d(embeddings_3d, labels)


model = MyModel()
checkpoint_path = "Out/onlyDifBlock.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])

df = pd.read_csv("src/data/Deng's dataset/3/test.csv")
df = df[df["label"].between(0, 4)]
sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 200)))
print(sampled_df.iloc[0])
sampled_df = [sampled_df.iloc[i] for i in range(len(sampled_df))]
print(sampled_df[0])

embeds, adjs, masks, cnn_masks, labels = collate_fn(sampled_df)

embeds = embeds.cpu()
adjs = adjs.cpu()
masks = masks.cpu()
cnn_masks = cnn_masks.cpu()
labels = labels.cpu().numpy()

print(model)

target_layer = model.mlp[0]

draw_umap(embeds, adjs, masks, cnn_masks,
          labels=labels,
          model=model,
          target_layer=target_layer
          )

