from utils import *

####

eval_image_name = "eval_samples_attr_embedding/00046_00000.tiff"

#####

spec = load_spectrogram_img(eval_image_name)

width = spec.shape[1]
original_spec = spec[:, :width//2]
reconstructed_spec = spec[:, width//2:]

original_sample = spectrogram_to_wav(original_spec)
original_sample = torch.unsqueeze(original_sample, dim=0)
save_wav_to_file(original_sample, "original.wav")

reconstructed_sample = spectrogram_to_wav(reconstructed_spec)
reconstructed_sample = torch.unsqueeze(reconstructed_sample, dim=0)
save_wav_to_file(reconstructed_sample, "reconstructed.wav")