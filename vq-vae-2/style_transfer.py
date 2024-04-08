from torch.utils.data import DataLoader
from dataset import LMDBDataset
from model_definitions import getVQVAE
import numpy as np
import torch
import torchaudio
from sample_with_prior import spectrogram_to_wav, GTZAN_SAMPLE_RATE
from torchvision.utils import save_image
from PIL import Image
import librosa
import soundfile as sf
import noisereduce as nr

def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).detach().cpu().numpy())
    img.save(path, format='TIFF')

embedding_path = "latent_embeddings_vanilla"

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
    
    
###########

dataset_raggae = LMDBDataset(embedding_path, desired_class_label=8)

loader = DataLoader(dataset_raggae, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

top_embeddings_raggae = []
bottom_embeddings_raggae = []
labels_raggae = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings_raggae.append(top)
    bottom_embeddings_raggae.append(bottom)
    labels_raggae.append(label.numpy().ravel())


###########
dataset_pop = LMDBDataset(embedding_path, desired_class_label=7)

loader = DataLoader(dataset_pop, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

top_embeddings_pop = []
bottom_embeddings_pop = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings_pop.append(top)
    bottom_embeddings_pop.append(bottom)
    
    
###########
dataset_jazz = LMDBDataset(embedding_path, desired_class_label=5)

loader = DataLoader(dataset_jazz, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

top_embeddings_jazz = []
bottom_embeddings_jazz = []

for i, (label, top, bottom, filename) in enumerate(loader):
    top_embeddings_jazz.append(top)
    bottom_embeddings_jazz.append(bottom)
    

## style transfer

vae = getVQVAE(False, device)

vae.load_state_dict(torch.load("checkpoints_vanilla/vqvae_200.pt"))

vae = vae.to(device)

shape_top = 64
shape_bottom = 128

half_height_top = int(shape_top*0.6)
half_height_bottom = int(shape_bottom*0.4)

first_embedding = (np.array(top_embeddings_pop[9]), np.array(bottom_embeddings_pop[9]))
second_embedding = (np.array(top_embeddings_raggae[5]), np.array(bottom_embeddings_raggae[5]))

top_half_1 = np.squeeze(first_embedding[0])[:half_height_top, :]
top_half_2 = np.squeeze(second_embedding[0])[half_height_top:, :]
combined_top = np.concatenate([top_half_1, top_half_2], axis=0)

bottom_half_1 = np.squeeze(first_embedding[1])[:half_height_bottom, :]
bottom_half_2 = np.squeeze(second_embedding[1])[half_height_bottom:, :]
combined_bottom = np.concatenate([bottom_half_1, bottom_half_2], axis=0)

top_embeddings_style = combined_top.reshape(1, shape_top, shape_top)
bottom_embeddings_style = combined_bottom.reshape(1, shape_bottom, shape_bottom)

decoded_sample = vae.decode_code(torch.tensor(top_embeddings_style, dtype=torch.int64).to(device), torch.tensor(bottom_embeddings_style, dtype=torch.int64).to(device))
decoded_sample = torch.exp(decoded_sample) - 1.0

save_spectrogram_img(decoded_sample, "generated_samples/style_transfer.tiff")

waveform = spectrogram_to_wav(decoded_sample.squeeze(0).detach())
torchaudio.save("generated_samples/style_transfer.wav", waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)

data, samplerate = sf.read("generated_samples/style_transfer.wav")
y_reduced_noise = nr.reduce_noise(y=data, sr=samplerate, prop_decrease=0.6)

sf.write('generated_samples/denoised_style_transfer.wav', y_reduced_noise, samplerate)
