import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GTZAN_SAMPLE_RATE = 22050
N_MELS = 512
N_FFTS = 2586
SPEC_TRANSFORM = torchaudio.transforms.MelSpectrogram(sample_rate=GTZAN_SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFTS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
def wav_to_spectrogram(sample):
    return SPEC_TRANSFORM(sample).to(DEVICE)
    
def spectrogram_to_wav(spectrogram):
    print("Inverting Mel Scale...")
    inverse_transform = torchaudio.transforms.InverseMelScale(n_mels=N_MELS, sample_rate=GTZAN_SAMPLE_RATE, n_stft=int((N_FFTS//2)+1)).to(DEVICE)
    spectrogram = inverse_transform(spectrogram)
    print("Converting to waveform...")
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=N_FFTS).to(DEVICE)
    return grifflim_transform(spectrogram)
    
def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).detach().cpu().numpy())
    img.save(path, format='TIFF')
    
def save_wav_to_file(sample, path):
    torchaudio.save(path, sample.detach().cpu(), sample_rate=GTZAN_SAMPLE_RATE)
    
def load_spectrogram_img(path):
    image = Image.open(path)
    return torch.tensor(np.array(image), requires_grad=False).to(DEVICE)
    
def load_wav_file(path):
    sample, _ = torchaudio.load(path)
    return sample

def computeSpectrogramsMinMax(folder_path):
    overall_max = 0
    overall_min = 9999
    for genre in GENRES:
        folder = folder_path + "/{}".format(genre)
        filenames = os.listdir(folder)
        for filename in filenames:
            spec = load_spectrogram_img(folder + "/" + filename)
            overall_max = max(torch.max(spec), overall_max)
            overall_min = min(torch.min(spec), overall_min)
            
            print(overall_max)
            print(overall_min)
            
def plot_spectrogram(spectrogram):
    plt.figure()
    # Compute the time axis
    num_frames = spectrogram.size(2)
    frame_length_sec = num_frames / spectrogram.size(1)  # Assuming each frame is 1 second

    # Plot the spectrogram
    p = plt.imshow(torch.log2(spectrogram[0, :, :]).numpy(), cmap='gray', aspect='auto', origin='lower',
                   extent=[0, frame_length_sec, 0, 8000])
    plt.colorbar(p, label='Intensity (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Mel Spectrogram')
    plt.show()
    
if __name__ == "__main__":
    # play a random spectrogram from the augmented dataset
    random_genre = GENRES[random.randint(0, 9)]
    random_i = random.randint(0, 800)
    random_spectrogram = "AugmentedGTZAN/{}/{}_{}.tiff".format(random_genre, random_genre, random_i)
    print("Loading {} sample as image...".format(random_genre))
    spectrogram = load_spectrogram_img(random_spectrogram)
    sample = spectrogram_to_wav(spectrogram)
    sample = torch.unsqueeze(sample, dim=0)
    print("Saving to file...")
    save_wav_to_file(sample, "temp.wav")