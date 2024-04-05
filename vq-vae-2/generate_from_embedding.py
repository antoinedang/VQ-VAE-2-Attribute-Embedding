from torch.utils.data import DataLoader
from dataset import LMDBDataset
from model_definitions import getVQVAE
import random
import torch
import torchaudio
from sample import spectrogram_to_wav, GTZAN_SAMPLE_RATE
from PIL import Image

def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).detach().cpu().numpy())
    img.save(path, format='TIFF')

embedding_path = "latent_embeddings_10"
genre = 1
model_ckpt = "checkpoints_10/vqvae_200.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LMDBDataset(embedding_path, desired_class_label=genre)

rand_index = random.randint(0, len(dataset)-1)
_, top, bottom, _ = dataset[rand_index]

vae = getVQVAE(False, device)
vae.load_state_dict(torch.load(model_ckpt))

decoded_sample = vae.decode_code(top.unsqueeze(0).to(device), bottom.unsqueeze(0).to(device))
decoded_sample = torch.exp(decoded_sample) - 1.0

save_spectrogram_img(decoded_sample, "generated_samples/from_embedding.tiff")

waveform = spectrogram_to_wav(decoded_sample.squeeze(0).detach(), device=device)
torchaudio.save("generated_samples/from_embedding.wav", waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)


