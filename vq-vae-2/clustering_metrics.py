from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from torch.utils.data import DataLoader
from dataset import LMDBDataset
import numpy as np
import matplotlib.pyplot as plt

embedding_paths = ["latent_embeddings_vanilla","latent_embeddings_vanilla_augmented","latent_embeddings_0.1", "latent_embeddings_1", "latent_embeddings_10"]

davies_top_list = []
silhouette_top_list = []
davies_bottom_list = []
silhouette_bottom_list = []
calinski_top_list = []
calinski_bottom_list = []

for path in embedding_paths:

    dataset = LMDBDataset(path, desired_class_label=None)

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
    davies_top_list.append(davies_bouldin_top)
    silhouette_top_list.append(silhouette_avg_top)
    davies_bottom_list.append(davies_bouldin_bottom)
    silhouette_bottom_list.append(silhouette_avg_bottom)
    calinski_top_list.append(calinski_harabasz_score_top)
    calinski_bottom_list.append(calinski_harabasz_score_bottom)

    print("The following results are for the embeddings: " + path)

    print("Davies Bouldin Score for Top Embeddings: ", davies_bouldin_top)
    print("Davies Bouldin Score for Bottom Embeddings: ", davies_bouldin_bottom)

    print("Silhouette Score for Top Embeddings: ", silhouette_avg_top)
    print("Silhouette Score for Bottom Embeddings: ", silhouette_avg_bottom)

    print("Calinski Harabasz Score for Top Embeddings: ", calinski_harabasz_score_top)
    print("Calinski Harabasz Score for Bottom Embeddings: ", calinski_harabasz_score_bottom)


weights = ["0", "0-augmented", "0.1", "1", "10"]
min_weight = min(weights)
max_weight = max(weights)

## Plotting the results

# Davies Bouldin Score
plt.plot(weights, davies_top_list, label='Davies Bouldin Score for Top Embeddings')
plt.plot(weights, davies_bottom_list, label='Davies Bouldin Score for Bottom Embeddings')
plt.xlabel('Weight of Custom Loss')
plt.ylabel('Davies Bouldin Score')
plt.title('Davies Bouldin Score for Top and Bottom Embeddings in terms of Weight of Custom Loss')
plt.legend()
plt.show()

# Silhouette Score
plt.plot(weights, silhouette_top_list, label='Silhouette Score for Top Embeddings')
plt.plot(weights, silhouette_bottom_list, label='Silhouette Score for Bottom Embeddings')
plt.xlabel('Weight of Custom Loss')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Top and Bottom Embeddings in terms of Weight of Custom Loss')
plt.legend()
plt.show()

# Calinski Harabasz Score
plt.plot(weights, calinski_top_list, label='Calinski Harabasz Score for Top Embeddings')
plt.plot(weights, calinski_bottom_list, label='Calinski Harabasz for Bottom Embeddings')
plt.xlabel('Weight of Custom Loss')
plt.ylabel('Calinski Harabasz Score')
plt.title('Calinski Harabasz Score for Top and Bottom Embeddings in terms of Weight of Custom Loss')
plt.legend()
plt.show()