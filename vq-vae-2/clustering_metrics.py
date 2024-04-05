from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from torch.utils.data import DataLoader
from dataset import LMDBDataset
import numpy as np

embedding_path = "latent_embeddings_10"

dataset = LMDBDataset(embedding_path, desired_class_label=None)

loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

top_embeddings = []
bottom_embeddings = []
labels = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings.append(top.squeeze().cpu().numpy().ravel())
    bottom_embeddings.append(bottom.squeeze().cpu().numpy().ravel())
    labels.append(label.numpy().ravel())

top_embeddings = np.array(top_embeddings)
bottom_embeddings = np.array(bottom_embeddings)
labels = np.array(labels).ravel()

davies_bouldin_top = davies_bouldin_score(top_embeddings, labels)
silhouette_avg_top = silhouette_score(top_embeddings, labels)
davies_bouldin_bottom = davies_bouldin_score(bottom_embeddings, labels)
silhouette_avg_bottom = silhouette_score(bottom_embeddings, labels)
calinski_harabasz_score_top = calinski_harabasz_score(top_embeddings, labels)
calinski_harabasz_score_bottom = calinski_harabasz_score(bottom_embeddings, labels)

print("The following results are for the embeddings: " + embedding_path)

print("Davies Bouldin Score for Top Embeddings: ", davies_bouldin_top)
print("Davies Bouldin Score for Bottom Embeddings: ", davies_bouldin_bottom)

print("Silhouette Score for Top Embeddings: ", silhouette_avg_top)
print("Silhouette Score for Bottom Embeddings: ", silhouette_avg_bottom)

print("Calinski Harabasz Score for Top Embeddings: ", calinski_harabasz_score_top)
print("Calinski Harabasz Score for Bottom Embeddings: ", calinski_harabasz_score_bottom)