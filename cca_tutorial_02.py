"""
Author: Velu Prabhakar Kumaravel
Affiliation: University of Oldenburg
Date: 13/08/2024
Contact: vpr.kumaravel@gmail.com

Description:
This script is an extension of cca_tutorial_01.py - precisely, we understand the EEG channel weights that maximized the correlation
with the sinusoidal reference signal. I plotted the weights as bar graphs and topo plots.

Dependencies:
- numpy
- matplotlib
- h5py
- sklearn
- scipy

Usage:
- Ensure that the required data file (`data_s19_64.mat`) is downloaded using:
https://springernature.figshare.com/collections/An_open_dataset_for_human_SSVEPs_in_the_frequency_range_of_1-60_Hz/6752910/1.
- Run the script to visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.cross_decomposition import CCA
import mne
import os

npy_file = 'data_s19_64.npy'
mat_file = 'data_s19_64.mat'

# Generate reference signals for 12 Hz
def generate_reference_signals(frequencies, num_samples, sample_rate, num_harmonics=1):
    t = np.arange(num_samples) / sample_rate
    references = []

    for f in frequencies:
        for n in range(1, num_harmonics + 1):  # Includes the fundamental and up to num_harmonics-1 harmonics
            references.append(np.sin(2 * np.pi * n * f * t))

    return np.array(references).T

# Load the data
if os.path.exists(npy_file):
    data = np.load(npy_file)
else:
    with h5py.File(mat_file, 'r') as mat_file:
        data = mat_file['datas'][:]  # Read the data into a NumPy array
    np.save(npy_file, data)

# Extract specific condition
condition_index = 0  # Low-Depth condition
frequency_index = 11  # 12 Hz frequency
block_index = 1  # block 2

# Extract the data for the given condition
X = data[block_index, frequency_index, :, :, condition_index]
X_reshaped = X.reshape(-1, 64)

# Parameters
frequencies = [12]  # 12 Hz
num_samples = X.shape[0]  # Number of time points
sample_rate = 1000

reference_signals = generate_reference_signals(frequencies, num_samples, sample_rate, num_harmonics=1)
print(np.median(reference_signals), np.median(X_reshaped))

cca = CCA(n_components=1)
cca.fit(reference_signals, X_reshaped)
cca_scores = cca.score(reference_signals, X_reshaped)
print("CCA Scores:", cca_scores)
X_c, Y_c = cca.transform(reference_signals, X_reshaped)

# Extract the canonical weights for the EEG channels
canonical_weights = cca.y_weights_  # This should be of shape (num_channels, n_components)

# Print the canonical weights
print("Canonical Weights:\n", canonical_weights)

# Flatten the weights if necessary for visualization
canonical_weights_flat = canonical_weights.flatten()

# Assuming the following montage for neuroscan!!
montage = mne.channels.make_standard_montage('standard_1020')
channels = [
    ('Fp1', -29, 104.536, 33.1255),
    ('Fpz', 1, 109.203, 38.6223),
    ('Fp2', 30, 104.651, 34.1187),
    ('AF3', -34, 101.932, 62.625),
    ('AF4', 36, 101.054, 63.734),
    ('F11', -74, 46.771, -30.6181),
    ('F7', -70, 61.8921, 30.0395),
    ('F5', -66, 68.4593, 60.4841),
    ('F3', -52, 76.7816, 88.7107),
    ('F1', -28, 84.0625, 107.998),
    ('Fz', 2, 88.7299, 113.494),
    ('F2', 31, 84.8243, 105.895),
    ('F4', 53, 78.6525, 87.4859),
    ('F6', 66, 69.684, 62.3549),
    ('F8', 70, 61.2459, 33.1351),
    ('F12', 74, 44.4855, -24.3113),
    ('FT11', -80, 18.5638, -39.413),
    ('FC5', -77, 39.5866, 71.902),
    ('FC3', -62, 47.61, 106.204),
    ('FC1', -34, 54.3605, 129.58),
    ('FCz', 3, 57.2728, 137.295),
    ('FC2', 37, 54.129, 127.593),
    ('FC4', 63, 48.3719, 104.102),
    ('FC6', 77, 40.6956, 72.7796),
    ('FT12', 79, 16.394, -32.1129),
    ('T7', -84, 3.51988, 38.8537),
    ('C5', -83, 8.38021, 80.5715),
    ('C3', -69, 14.7642, 118.085),
    ('C1', -37, 18.882, 144.788),
    ('Cz', 3, 22.0258, 154.489),
    ('C2', 40, 19.6438, 142.685),
    ('C4', 69, 15.7575, 117.969),
    ('C6', 83, 9.48921, 81.4491),
    ('T8', 84, 3.98277, 42.8268),
    ('TP7', -82, -26.0471, 44.3119),
    ('CP5', -81, -25.3914, 84.5061),
    ('CP3', -67, -22.9805, 122.482),
    ('CP1', -36, -20.8493, 149.417),
    ('CPz', 3, -17.8212, 158.125),
    ('CP2', 40, -18.1009, 147.083),
    ('CP4', 67, -19.0074, 122.019),
    ('CP6', 80, -21.1868, 86.0298),
    ('TP8', 82, -24.591, 48.1693),
    ('M1', -76, -41.2549, 8.83346),
    ('M2', 77, -38.4583, 15.555),
    ('P7', -73, -54.5051, 50.6477),
    ('P5', -68, -55.6527, 84.0046),
    ('P3', -56, -55.1609, 114.15),
    ('P1', -30, -54.7173, 135.241),
    ('Pz', 2, -52.9139, 142.078),
    ('P2', 32, -52.7307, 135.009),
    ('P4', 56, -53.1743, 113.919),
    ('P6', 68, -52.5571, 84.6507),
    ('P8', 72, -52.1714, 53.3961),
    ('PO7', -54, -77.6496, 59.3847),
    ('PO3', -37, -79.559, 94.8439),
    ('POz', 2, -78.7007, 110.852),
    ('PO4', 37, -78.6814, 93.7349),
    ('PO8', 55, -75.5473, 60.1466),
    ('O1', -29, -91.8544, 67.0803),
    ('Oz', 1, -94.0242, 74.3804),
    ('O2', 30, -90.7454, 67.9578),
    ('Cb1', -28, -92.8091, 32.9615),
    ('Cb2', 31, -90.7069, 33.7234),
]

labels, xs, ys, zs = zip(*channels)
coords = np.array([xs, ys, zs]).T

info = mne.create_info(
    ch_names=labels,
    ch_types='eeg',
    sfreq=1000
)

montage = mne.channels.make_dig_montage(
    ch_pos={label: (x, y, z) for label, x, y, z in channels}
)
info.set_montage(montage)

data = np.zeros((len(canonical_weights_flat), 1))
raw = mne.io.RawArray(data, info)
raw._data = canonical_weights_flat[:, np.newaxis]
print(raw._data)

channel_names = [ch[0] for ch in channels]
plt.figure(figsize=(12, 6))
plt.bar(channel_names, canonical_weights_flat, color='skyblue')
plt.xlabel('Channel Name')
plt.ylabel('Canonical Weight')
plt.title('Canonical Weights for Each EEG Channel')
plt.grid(True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('figures/Fig04_CCA_Weights_Bar.png', dpi=300)
plt.show()

fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(canonical_weights_flat,
                     info,
                     axes=ax,
                     image_interp="cubic",
                     names=channel_names,
                              show=False)
ax_x_start = 0.85
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.75
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title("Canonical Weights")
plt.savefig('figures/Fig05_CCA_Weights_Topo.png', dpi=300)
plt.show()