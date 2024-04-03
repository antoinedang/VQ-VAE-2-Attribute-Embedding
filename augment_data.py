import torch
import torchaudio
import os
from utils import *

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
GTZAN_SAMPLE_RATE = 22050
WAVEFORM_LENGTH = 30 * 22050 # 30 seconds

gain_transforms = [torchaudio.transforms.Vol(g) for g in [1, 0.5, 1.5]]
pitch_transforms = [torchaudio.transforms.PitchShift(GTZAN_SAMPLE_RATE, n) for n in [0, -2, 2]]

total_augmented_samples = 0
iterations_completed = 0
total_iterations_required = 1000 * len(gain_transforms) * len(pitch_transforms) # 1000 since GTZAN starts with 1000 samples

if not os.path.exists("AugmentedGTZAN"): os.makedirs("AugmentedGTZAN")

print("Augmenting to {} samples...".format(total_iterations_required))

for genre in GENRES:
  output_folder = "AugmentedGTZAN/{}".format(genre)
  if not os.path.exists(output_folder): os.makedirs(output_folder)
  i = 0
  raw_data_filenames = os.listdir("GTZAN/{}".format(genre))
  for raw_data_filename in raw_data_filenames:
    try:
      raw_sample = load_wav_file("GTZAN/{}".format(genre) + "/" + raw_data_filename)
    except:
      continue
    for gain_transform in gain_transforms:
      for pitch_shift in pitch_transforms:
        output_filename = output_folder + "/{}_{}.tiff".format(genre, i)
        # if os.path.exists(output_filename):
        #   i += 1
        #   total_augmented_samples += 1
        #   iterations_completed += 1
        #   continue
          
        gain_augmented_sample = gain_transform(raw_sample)
        full_augmented_sample = pitch_shift(gain_augmented_sample)
        sample = full_augmented_sample[:, :WAVEFORM_LENGTH]
        
        spectrogram = wav_to_spectrogram(sample)
        
        if spectrogram.shape[1] != N_MELS or spectrogram.shape[2] != N_MELS:
          print("Error: invalid shape output {}".format(spectrogram.shape))
          iterations_completed += 1
          continue
        
        save_spectrogram_img(spectrogram, output_filename)

        total_augmented_samples += 1
        i += 1
        iterations_completed += 1
        print("{}%             ".format(100 * iterations_completed / total_iterations_required), end='\r')
    
print("Augmentation successful. Increased dataset size from 1000 to {}".format(total_augmented_samples))