import torch
from torch.utils.data import Dataset
import torchaudio
import matplotlib.pyplot as plt
import os

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GTZAN_SAMPLE_RATE = 22050
WAVEFORM_LENGTH = 30 * 22050 # 30 seconds
SPEC_TRANSFORM = torchaudio.transforms.MelSpectrogram(GTZAN_SAMPLE_RATE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device: {}".format(DEVICE))
        
def load_spectrogram_from_wav(file_path):
    sample, _ = torchaudio.load(file_path)
    sample = sample[:, :WAVEFORM_LENGTH]
    sample = SPEC_TRANSFORM(sample).to(DEVICE)
    
def save_wav_from_spectrogram(file_path, spectrogram):
    waveform = torchaudio.transforms.InverseMelScale()(spectrogram)
    waveform = torchaudio.transforms.GriffinLim()(waveform)
    torchaudio.save(file_path, waveform, sample_rate=GTZAN_SAMPLE_RATE)
    
def plot_spectrogram(spectrogram):
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram.squeeze().detach().cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.show()
     
class LazyGTZANDataset(Dataset):
    def __init__(self, dataset_folder="AugmentedGTZAN"):
        self.file_paths = []
        for genre in GENRES:
            genre_filepaths = ["{}/{}/{}".format(dataset_folder, genre, fname) for fname in os.listdir("{}/{}".format(dataset_folder, genre))]
            self.file_paths.extend(genre_filepaths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = load_spectrogram_from_wav(file_path)
        
        for i in range(len(GENRES)):
            if GENRES[i] in file_path:
                label = i
        
        return sample, label
