from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = "eval_sample_1"

max_images = 200

mse_list =[]
epochs = []

for x in range(1, max_images + 1):

    filename = f'{x:05d}_00000.tiff'

    image_path = os.path.join(folder_path, filename)

    spec = load_spectrogram_img(image_path).detach().cpu()
    width = spec.shape[1]
    original_spec = np.array(spec[:, :width//2])
    reconstructed_spec = np.array(spec[:, width//2:])

    mse = np.mean((np.log(original_spec+1) - np.log(reconstructed_spec+1)) ** 2)

    mse_list.append(mse)
    epochs.append(x)

# plt.plot(epochs)
plt.plot(mse_list)

plt.xlabel('Epoch Number')
plt.ylabel('Normalized MSE Value')
plt.title('Normalized MSE Value in terms of epoch number')

plt.show()








