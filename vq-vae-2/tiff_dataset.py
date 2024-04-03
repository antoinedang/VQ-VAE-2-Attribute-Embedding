import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset


GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def load_spectrogram_img(path):
    image = Image.open(path)
    return torch.tensor(np.array(image))

class TIFFDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        for i in range(len(GENRES)):
            filenames = os.listdir(folder_path + "/{}".format(GENRES[i]))
            for filename in filenames:
                self.samples.append((folder_path + "/{}/{}".format(GENRES[i], filename), torch.tensor(i)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = load_spectrogram_img(image_path)
        return torch.unsqueeze(torch.log(image + 1.0), dim=0), label