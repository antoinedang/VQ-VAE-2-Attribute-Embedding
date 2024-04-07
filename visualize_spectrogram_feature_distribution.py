from utils import *

####
max_images = 200
eval_folder = "eval_samples_vanilla"

all_feature_values = []

for x in range(1, max_images + 1):

    train_filename = f'{x:05d}_00000_train.tiff'
    test_filename = f'{x:05d}_00000_test.tiff'

    train_image_path = os.path.join(eval_folder, train_filename)
    test_image_path = os.path.join(eval_folder, test_filename)
    try:
        train_spec = load_spectrogram_img(train_image_path)
        width = train_spec.shape[1]
        original_train_spec = train_spec[:, :width//2].flatten().detach().cpu().numpy()
        test_spec = load_spectrogram_img(test_image_path)
        width = test_spec.shape[1]
        original_test_spec = test_spec[:, :width//2].flatten().detach().cpu().numpy()
    except:
        continue
    
    all_feature_values.extend(original_test_spec)
    all_feature_values.extend(original_train_spec)

#####

all_feature_values = np.log(np.array(all_feature_values) + 1.0)

num_bins = 100
hist, bins = np.histogram(all_feature_values, bins=num_bins)
min_value = np.min(all_feature_values)
max_value = np.max(all_feature_values)

# Normalize the histogram to get relative frequencies
hist = hist / len(all_feature_values)

# Plot the histogram
plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]))
# plt.xscale('log')
plt.yscale('log')
plt.xlabel('Feature Value')
plt.ylabel('Relative Frequency')
plt.axvline(min_value, color='r', linestyle='--', label='Min Value')
plt.axvline(max_value, color='g', linestyle='--', label='Max Value')
plt.title('Distribution of Feature Values in Spectrograms\n(with Normalization through logarithmic scaling)')
plt.legend()
plt.show()