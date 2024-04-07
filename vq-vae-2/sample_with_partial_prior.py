import argparse
import os

import torch
from tqdm import tqdm
import torchaudio
import numpy as np
from PIL import Image

from model_definitions import *
from dataset import LMDBDataset
from tiff_dataset import GENRES
import random

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, genre, device, checkpoint_folder):
    all_checkpoint_filenames = os.listdir(checkpoint_folder)
    highest_epoch_checkpoint = None
    highest_epoch_num = -1
    for checkpoint in all_checkpoint_filenames:
        if model != 'vqvae' and "vae" in checkpoint: continue
        if model == 'pixelsnail_bottom' and "top" in checkpoint: continue
        if model == 'pixelsnail_top' and "bottom" in checkpoint: continue
        if (model != 'vqvae') and (genre not in checkpoint): continue
        epoch_num = int(checkpoint.split(".")[0].split("_")[-1])
        if epoch_num > highest_epoch_num: highest_epoch_checkpoint = checkpoint
        
    ckpt = torch.load(os.path.join(checkpoint_folder, highest_epoch_checkpoint))
    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = getVQVAE(embed_labels=False, device=device)

    elif model == 'pixelsnail_top':
        model = getPixelSnailTop()

    elif model == 'pixelsnail_bottom':
        model = getPixelSnailBottom()
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu": print("WARN: CUDA not available. Training will take very long.")
GTZAN_SAMPLE_RATE = 22050
N_MELS = 512
N_FFTS = 2586

def spectrogram_to_wav(spectrogram):
    print("Inverting Mel Scale...")
    inverse_transform = torchaudio.transforms.InverseMelScale(n_mels=N_MELS, sample_rate=GTZAN_SAMPLE_RATE, n_stft=int((N_FFTS//2)+1)).to(device)
    spectrogram = inverse_transform(spectrogram)
    print("Converting to waveform...")
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=N_FFTS).to(device)
    return grifflim_transform(spectrogram)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--genre', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--checkpoint-folder', type=str, default="checkpoints")
    parser.add_argument('--num-iterations', type=int, default=1)
    parser.add_argument('--embeddings', type=str, default="latent_embeddings")
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', genre=None, device=device, checkpoint_folder=args.checkpoint_folder)
    model_bottom = load_model('pixelsnail_bottom', genre=args.genre, device=device, checkpoint_folder=args.checkpoint_folder)

    dataset = LMDBDataset(args.embeddings, desired_class_label=GENRES.index(args.genre))

    rand_index = random.randint(0, len(dataset)-1)
    top_sample = []
    for _ in range(args.batch):
        _, top, _, _ = dataset[rand_index]
        top_sample.append(top)
    top_sample = torch.tensor(top_sample).to(device)
        
    bottom_sample = sample_model(model_bottom, device, args.batch, [128, 128], args.temp, condition=top_sample)
    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    
    # https://arxiv.org/abs/1610.09296
    for _ in tqdm(range(args.num_iterations-1)):
        decoded_sample, _, _ = model_vqvae(decoded_sample, 0)
    
    for i in range(decoded_sample.shape[0]):
        generated_spec = torch.tensor(np.exp(decoded_sample[i].detach().cpu().numpy()) - 1.0).to(device)
        img = Image.fromarray(torch.squeeze(generated_spec).detach().cpu().numpy())
        img.save("generated_samples/" + args.filename + "_" + str(i+1) + ".tiff", format='TIFF')
        waveform = spectrogram_to_wav(generated_spec.detach())
        torchaudio.save("generated_samples/" + args.filename + "_" + str(i+1) + ".wav", waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)
