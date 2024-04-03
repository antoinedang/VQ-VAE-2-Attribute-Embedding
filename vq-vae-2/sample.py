import argparse
import os

import torch
from tqdm import tqdm
import torchaudio

from model_definitions import *

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


def load_model(model, genre, device):
    all_checkpoint_filenames = os.listdir("checkpoints")
    highest_epoch_checkpoint = None
    highest_epoch_num = -1
    for checkpoint in all_checkpoint_filenames:
        if model != 'vqvae' and "vae" in checkpoint: continue
        if model == 'pixelsnail_bottom' and "top" in checkpoint: continue
        if model == 'pixelsnail_top' and "bottom" in checkpoint: continue
        if (model != 'vqvae') and (genre not in checkpoint): continue
        epoch_num = int(checkpoint.split(".")[0].split("_")[-1])
        if epoch_num > highest_epoch_num: highest_epoch_checkpoint = checkpoint
        
    ckpt = torch.load(os.path.join('checkpoints', highest_epoch_checkpoint))
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARN: CUDA not available. Training will take very long.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--genre', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', genre=None, device=device)
    model_top = load_model('pixelsnail_top', genre=args.genre, device=device)
    model_bottom = load_model('pixelsnail_bottom', genre=args.genre, device=device)

    top_sample = sample_model(model_top, device, args.batch, [32*2, 32*2], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [64*2, 64*2], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    
    for i in range(decoded_sample.shape[0]):
        generated_spec = torch.exp(decoded_sample[i]) - 1.0
        waveform = spectrogram_to_wav(generated_spec.detach())   
        torchaudio.save("generated_samples/" + args.filename, waveform.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)
