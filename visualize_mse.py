from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

folder_paths = [("eval_samples_0.1", "Margin Loss Weight 0.1"),
                ("eval_samples_1", "Margin Loss Weight 1"),
                ("eval_samples_10", "Margin Loss Weight 10"),
                ("eval_samples_vanilla", "Margin Loss Weight 0")]

max_images = 200

for folder_path, label in folder_paths:
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
    plt.plot(epochs, mse_list, label=label)

plt.xlabel('# Epochs')
plt.ylabel('MSE')

plt.legend()
plt.show()