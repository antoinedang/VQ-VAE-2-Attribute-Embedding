from torch.utils.data import DataLoader
from dataset import LMDBDataset
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from tiff_dataset import GENRES

embedding_path = "latent_embeddings_10"

def visualize_TSNE():
    dataset = LMDBDataset(embedding_path, desired_class_label=None)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    top_embeddings = []
    bottom_embeddings = []
    labels = []

    for i, (label, top, bottom, filename) in enumerate(loader):
        top_embeddings.append(top.squeeze().cpu().numpy())
        bottom_embeddings.append(bottom.squeeze().cpu().numpy())
        labels.append(label)

    labels = np.array(labels).reshape(998)
    top_embeddings = np.array(top_embeddings).reshape(-1, 64*64)
    bottom_embeddings = np.array(bottom_embeddings).reshape(-1, 128*128)

    # TOP
    embeddings_reduced = TSNE(n_components=3, perplexity=100).fit_transform(top_embeddings) 

    top_X = embeddings_reduced[:, 0]
    top_Y = embeddings_reduced[:, 1]
    top_Z = embeddings_reduced[:, 2]

    colors = ['#1a85ff', '#ff9933', '#33cc33', '#ff3333', '#b366ff', '#996633', '#ff66cc', '#737373', '#99cc00', '#00b3e6']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(top_X), max(top_X))
    ax.set_ylim(min(top_Y), max(top_Y))
    ax.set_zlim(min(top_Z), max(top_Z))
    for i in range(10):
        ax.scatter(top_X[labels == i], top_Y[labels == i], top_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Top Latent Space')
    ax.legend()
    plt.show()

    # BOTTOM
    embeddings_reduced = TSNE(n_components=3, perplexity=100).fit_transform(bottom_embeddings) 

    bottom_X = embeddings_reduced[:, 0]
    bottom_Y = embeddings_reduced[:, 1]
    bottom_Z = embeddings_reduced[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(bottom_X), max(bottom_X))
    ax.set_ylim(min(bottom_Y), max(bottom_Y))
    ax.set_zlim(min(bottom_Z), max(bottom_Z))

    for i in range(10):
        ax.scatter(bottom_X[labels == i], bottom_Y[labels == i], bottom_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Bottom Latent Space')
    ax.legend()
    plt.show()

def visualize_LLE():
    dataset = LMDBDataset(embedding_path, desired_class_label=None)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    top_embeddings = []
    bottom_embeddings = []
    labels = []

    for i, (label, top, bottom, filename) in enumerate(loader):
        top_embeddings.append(top.squeeze().cpu().numpy())
        bottom_embeddings.append(bottom.squeeze().cpu().numpy())
        labels.append(label)

    labels = np.array(labels).reshape(998)
    top_embeddings = np.array(top_embeddings).reshape(-1, 64*64)
    bottom_embeddings = np.array(bottom_embeddings).reshape(-1, 128*128)

    # TOP
    embeddings_reduced = LocallyLinearEmbedding(n_components=3, n_neighbors=50).fit_transform(top_embeddings)

    top_X = embeddings_reduced[:, 0]
    top_Y = embeddings_reduced[:, 1]
    top_Z = embeddings_reduced[:, 2]

    colors = ['#1a85ff', '#ff9933', '#33cc33', '#ff3333', '#b366ff', '#996633', '#ff66cc', '#737373', '#99cc00', '#00b3e6']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(top_X), max(top_X))
    ax.set_ylim(min(top_Y), max(top_Y))
    ax.set_zlim(min(top_Z), max(top_Z))
    for i in range(10):
        ax.scatter(top_X[labels == i], top_Y[labels == i], top_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Top Latent Space')
    ax.legend()
    plt.show()

    # BOTTOM
    embeddings_reduced = LocallyLinearEmbedding(n_components=3, n_neighbors=100).fit_transform(bottom_embeddings)

    bottom_X = embeddings_reduced[:, 0]
    bottom_Y = embeddings_reduced[:, 1]
    bottom_Z = embeddings_reduced[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(bottom_X), max(bottom_X))
    ax.set_ylim(min(bottom_Y), max(bottom_Y))
    ax.set_zlim(min(bottom_Z), max(bottom_Z))

    for i in range(10):
        ax.scatter(bottom_X[labels == i], bottom_Y[labels == i], bottom_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Bottom Latent Space')
    ax.legend()
    plt.show()

def visualize_SpectralEmbed():
    dataset = LMDBDataset(embedding_path, desired_class_label=None)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    top_embeddings = []
    bottom_embeddings = []
    labels = []

    for i, (label, top, bottom, filename) in enumerate(loader):
        top_embeddings.append(top.squeeze().cpu().numpy())
        bottom_embeddings.append(bottom.squeeze().cpu().numpy())
        labels.append(label)

    labels = np.array(labels).reshape(998)
    top_embeddings = np.array(top_embeddings).reshape(-1, 64*64)
    bottom_embeddings = np.array(bottom_embeddings).reshape(-1, 128*128)

    # TOP
    embeddings_reduced = SpectralEmbedding(n_components=3).fit_transform(top_embeddings)

    top_X = embeddings_reduced[:, 0]
    top_Y = embeddings_reduced[:, 1]
    top_Z = embeddings_reduced[:, 2]

    colors = ['#1a85ff', '#ff9933', '#33cc33', '#ff3333', '#b366ff', '#996633', '#ff66cc', '#737373', '#99cc00', '#00b3e6']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(top_X), max(top_X))
    ax.set_ylim(min(top_Y), max(top_Y))
    ax.set_zlim(min(top_Z), max(top_Z))
    for i in range(10):
        ax.scatter(top_X[labels == i], top_Y[labels == i], top_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Top Latent Space')
    ax.legend()
    plt.show()

    # BOTTOM
    embeddings_reduced = SpectralEmbedding(n_components=3).fit_transform(bottom_embeddings)

    bottom_X = embeddings_reduced[:, 0]
    bottom_Y = embeddings_reduced[:, 1]
    bottom_Z = embeddings_reduced[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(bottom_X), max(bottom_X))
    ax.set_ylim(min(bottom_Y), max(bottom_Y))
    ax.set_zlim(min(bottom_Z), max(bottom_Z))

    for i in range(10):
        ax.scatter(bottom_X[labels == i], bottom_Y[labels == i], bottom_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    # ax.set_title('Scatter Plot of Bottom Latent Space')
    ax.legend()
    plt.show()

def visualize_PCA():
    dataset = LMDBDataset(embedding_path, desired_class_label=None)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    top_embeddings = []
    bottom_embeddings = []
    labels = []

    for i, (label, top, bottom, filename) in enumerate(loader):
        top_embeddings.append(top.squeeze().cpu().numpy())
        bottom_embeddings.append(bottom.squeeze().cpu().numpy())
        labels.append(label)

    labels = np.array(labels).reshape(998)
    top_embeddings = np.array(top_embeddings).reshape(-1, 64*64)
    bottom_embeddings = np.array(bottom_embeddings).reshape(-1, 128*128)
    concat_embeddings = concat_embeddings = np.hstack((top_embeddings, bottom_embeddings))

    # CONCAT
    embeddings_reduced = PCA(n_components=3).fit_transform(concat_embeddings) 

    top_X = embeddings_reduced[:, 0]
    top_Y = embeddings_reduced[:, 1]
    top_Z = embeddings_reduced[:, 2]

    colors = ['#1a85ff', '#ff9933', '#33cc33', '#ff3333', '#b366ff', '#996633', '#ff66cc', '#737373', '#99cc00', '#00b3e6']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min(top_X), max(top_X))
    ax.set_ylim(min(top_Y), max(top_Y))
    ax.set_zlim(min(top_Z), max(top_Z))
    for i in range(10):
        ax.scatter(top_X[labels == i], top_Y[labels == i], top_Z[labels == i], color=colors[i], marker='.', label=GENRES[i])
    #ax.set_title('Scatter Plot of Concatenated Latent Space')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    # visualize_TSNE()
    # visualize_LLE()
    visualize_SpectralEmbed()