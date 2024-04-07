from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

bottom_csv_path = "checkpoints_vanilla/pixelsnail_classical_bottom_metrics.csv"
top_csv_path = "checkpoints_vanilla/pixelsnail_classical_top_metrics.csv"
label = "Margin Loss Weight 0"

# bottom_csv_path = "checkpoints_vanilla_augmented/pixelsnail_classical_bottom_metrics.csv"
# top_csv_path = "checkpoints_vanilla_augmented/pixelsnail_classical_top_metrics.csv"
# label = "Margin Loss Weight 0 w/ Augmented Data"

# bottom_csv_path = "checkpoints_0.1/pixelsnail_classical_bottom_metrics.csv"
# top_csv_path = "checkpoints_0.1/pixelsnail_classical_top_metrics.csv"
# label = "Margin Loss Weight 0.1"

# bottom_csv_path = "checkpoints_1/pixelsnail_classical_bottom_metrics.csv"
# top_csv_path = "checkpoints_1/pixelsnail_classical_top_metrics.csv"
# label = "Margin Loss Weight 1"

# bottom_csv_path = "checkpoints_10/pixelsnail_classical_bottom_metrics.csv"
# top_csv_path = "checkpoints_10/pixelsnail_classical_top_metrics.csv"
# label = "Margin Loss Weight 10"

bottom_train_loss_list = []
bottom_test_loss_list = []
bottom_train_acc_list = []
bottom_test_acc_list = []
top_train_loss_list = []
top_test_loss_list = []
top_train_acc_list = []
top_test_acc_list = []
epochs = []

with open(top_csv_path, 'r') as file_top, open(bottom_csv_path, 'r') as file_bottom:
    while True:
        top_line = file_top.readline()
        bottom_line = file_bottom.readline()
        if not top_line or not bottom_line:
            break
        if top_line.split(",")[0] == "epoch" or bottom_line.split(",")[0] == "epoch":
            continue
        
        _, avg_test_loss, avg_test_acc, avg_train_loss, avg_train_acc = top_line.split(",")
        
        top_train_loss_list.append(float(avg_train_loss))
        top_test_loss_list.append(float(avg_test_loss))
        top_train_acc_list.append(float(avg_train_acc))
        top_test_acc_list.append(float(avg_test_acc))
        
        epoch, avg_test_loss, avg_test_acc, avg_train_loss, avg_train_acc = bottom_line.split(",")
        epochs.append(epoch)
        bottom_train_loss_list.append(float(avg_train_loss))
        bottom_test_loss_list.append(float(avg_test_loss))
        bottom_train_acc_list.append(float(avg_train_acc))
        bottom_test_acc_list.append(float(avg_test_acc))

# plt.plot(epochs)
plt.plot(epochs, bottom_train_acc_list, label="Bottom Train Acc")
plt.plot(epochs, bottom_test_acc_list, label="Bottom Test Acc")
plt.plot(epochs, top_train_acc_list, label="Top Train Acc")
plt.plot(epochs, top_test_acc_list, label="Top Test Acc")

plt.xlabel('# Epochs')

plt.title(label)

plt.legend()
plt.show()


# plt.plot(epochs)
plt.plot(epochs, bottom_train_loss_list, label="Bottom Train Loss")
plt.plot(epochs, bottom_test_loss_list, label="Bottom Test Loss")
plt.plot(epochs, top_train_loss_list, label="Top Train Loss")
plt.plot(epochs, top_test_loss_list, label="Top Test Loss")

plt.xlabel('# Epochs')

plt.title(label)

plt.legend()
plt.show()