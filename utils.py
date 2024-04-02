import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import Audio
import random

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GTZAN_SAMPLE_RATE = 22050
N_MELS = 1024
N_FFTS = 1293
SPEC_TRANSFORM = torchaudio.transforms.MelSpectrogram(sample_rate=GTZAN_SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFTS)
    
def wav_to_spectrogram(sample):
    return SPEC_TRANSFORM(sample).detach()
    
def spectrogram_to_wav(spectrogram):
    print("Inverting Mel Scale...")
    inverse_transform = torchaudio.transforms.InverseMelScale(n_mels=N_MELS, sample_rate=GTZAN_SAMPLE_RATE, n_stft=int((N_FFTS//2)+1))
    spectrogram = inverse_transform(spectrogram)
    print("Converting to waveform...")
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=N_FFTS)
    return grifflim_transform(spectrogram)
    
def save_spectrogram_img(spectrogram, path):
    img = Image.fromarray(torch.squeeze(spectrogram).numpy())
    img.save(path, format='TIFF')
    
def save_wav_to_file(sample, path):
    torchaudio.save(path, sample, sample_rate=GTZAN_SAMPLE_RATE)
    
def load_spectrogram_img(path):
    image = Image.open(path)
    return torch.tensor(np.array(image), requires_grad=False)
    
def load_wav_file(path):
    sample, _ = torchaudio.load(path)
    return sample
    
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
    print("Loading image...")
    spectrogram = load_spectrogram_img(random_spectrogram)
    print("Starting conversion. (takes a while)")
    sample = spectrogram_to_wav(spectrogram)
    sample = torch.unsqueeze(sample, dim=0)
    print("Saving to file...")
    torchaudio.save("temp.wav", sample, sample_rate=GTZAN_SAMPLE_RATE)