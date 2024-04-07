from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

folder_paths = [
                ("eval_samples_vanilla", "Margin Loss Weight 0"),
                ("eval_samples_vanilla_augmented", "Margin Loss Weight 0 w/ Augmented Data"),
                # ("eval_samples_0.1", "Margin Loss Weight 0.1"),
                # ("eval_samples_1", "Margin Loss Weight 1"),
                # ("eval_samples_10", "Margin Loss Weight 10"),
                ]

max_images = 200

for folder_path, label in folder_paths:
    train_mse_list = []
    test_mse_list = []
    epochs = []

    for x in range(1, max_images + 1):

        train_filename = f'{x:05d}_00000_train.tiff'
        test_filename = f'{x:05d}_00000_test.tiff'

        train_image_path = os.path.join(folder_path, train_filename)
        test_image_path = os.path.join(folder_path, test_filename)

        train_spec = load_spectrogram_img(train_image_path).detach().cpu()
        test_spec = load_spectrogram_img(test_image_path).detach().cpu()
        
        width = train_spec.shape[1]
        original_spec = np.array(train_spec[:, :width//2])
        reconstructed_spec = np.array(train_spec[:, width//2:])

        train_mse = np.mean((np.log(original_spec+1) - np.log(reconstructed_spec+1)) ** 2)
        
        width = test_spec.shape[1]
        original_spec = np.array(test_spec[:, :width//2])
        reconstructed_spec = np.array(test_spec[:, width//2:])

        test_mse = np.mean((np.log(original_spec+1) - np.log(reconstructed_spec+1)) ** 2)

        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
        epochs.append(x)

    # plt.plot(epochs)
    plt.plot(epochs, train_mse_list, label="Train")
    plt.plot(epochs, test_mse_list, label="Test")

    plt.xlabel('# Epochs')
    plt.ylabel('MSE')
    
    plt.title(label)

    plt.legend()
    plt.show()