from torch.utils.data import DataLoader
from dataset import LMDBDataset
from model_definitions import getVQVAE
import numpy as np
import torch
import torchaudio
from sample import spectrogram_to_wav, GTZAN_SAMPLE_RATE
from torchvision.utils import save_image
from PIL import Image

def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).detach().cpu().numpy())
    img.save(path, format='TIFF')

embedding_path = "latent_embeddings_1"

dataset_classical = LMDBDataset(embedding_path, desired_class_label=1)

loader = DataLoader(dataset_classical, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

top_embeddings_classical = []
bottom_embeddings_classical = []
labels_classical = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings_classical.append(top)
    bottom_embeddings_classical.append(bottom)
    labels_classical.append(label.numpy().ravel())

# top_embeddings_classical = np.array(top_embeddings_classical)
# bottom_embeddings_classical = np.array(bottom_embeddings_classical)
labels_classical = np.array(labels_classical).ravel()

dataset_raggae = LMDBDataset(embedding_path, desired_class_label=8)

loader = DataLoader(dataset_raggae, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

top_embeddings_raggae = []
bottom_embeddings_raggae = []
labels_raggae = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings_raggae.append(top)
    bottom_embeddings_raggae.append(bottom)
    labels_raggae.append(label.numpy().ravel())

# top_embeddings_raggae = np.array(top_embeddings_raggae)
# bottom_embeddings_raggae = np.array(bottom_embeddings_raggae)
labels_raggae = np.array(labels_raggae).ravel()


vae = getVQVAE(False, device)

vae.load_state_dict(torch.load("checkpoints_1/vqvae_200.pt"))

vae = vae.to(device)

decoded_sample = vae.decode_code(top_embeddings_classical[0].to(device), bottom_embeddings_classical[0].to(device))
decoded_sample = torch.exp(decoded_sample) - 1.0
#decoded_sample = decoded_sample.clamp(-1, 1)

save_spectrogram_img(decoded_sample, "generated_samples/image1.tiff")

waveform = spectrogram_to_wav(decoded_sample.squeeze(0).detach(), device=device)
torchaudio.save("generated_samples/classical_from_code.wav", waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)


