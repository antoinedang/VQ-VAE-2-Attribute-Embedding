from torch.utils.data import DataLoader
from dataset import LMDBDataset
from model_definitions import getVQVAE
import random
import torch
import torchaudio
from sample_with_prior import spectrogram_to_wav, GTZAN_SAMPLE_RATE
from tiff_dataset import GENRES
from PIL import Image

def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).detach().cpu().numpy())
    img.save(path, format='TIFF')

embedding_path = "latent_embeddings_10"
genre = 0
model_ckpt = "checkpoints_10/vqvae_200.pt"
different_top_bottom = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LMDBDataset(embedding_path, desired_class_label=genre)

rand_index = random.randint(0, len(dataset)-1)
_, top, _, _ = dataset[rand_index]
print("TOP INDEX", rand_index)

if different_top_bottom: rand_index = random.randint(0, len(dataset)-1)
_, _, bottom, _ = dataset[rand_index]
print("BOTTOM INDEX", rand_index)

vae = getVQVAE(False, device)
vae.load_state_dict(torch.load(model_ckpt))

decoded_sample = vae.decode_code(top.unsqueeze(0).to(device), bottom.unsqueeze(0).to(device))
decoded_sample = torch.exp(decoded_sample) - 1.0

if different_top_bottom: filename = "{}_diff_top_bottom_embedding".format(GENRES[genre])
else: filename = "{}_from_embedding".format(GENRES[genre])

save_spectrogram_img(decoded_sample, "generated_samples/" + filename + ".tiff")

waveform = spectrogram_to_wav(decoded_sample.squeeze(0).detach())
torchaudio.save("generated_samples/" + filename + ".wav", waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)


