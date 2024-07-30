"""
Author: Velu Prabhakar Kumaravel
Affiliation: University of Oldenburg
Date: 30/07/2024
Contact: vpr.kumaravel@gmail.com

Description:
This script is designed for analyzing EEG data using Canonical Correlation Analysis (CCA).
It includes functionalities for loading data, generating reference signals,
and applying CCA to extract/visualize components and compute correlation scores for different reference frequencies.

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
from scipy.signal import welch
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
    with h5py.File(mat_file, 'r') as mat_file: # this method is time exhaustive; hence save the result as npy
        data = mat_file['datas'][:]  
    np.save(npy_file, data)

# Extract specific condition
condition_index = 0  # Low-Depth condition
frequency_index = 11  # 12 Hz frequency
block_index = 1  # block 2

# Extract the data for the given condition
X = data[block_index, frequency_index, :, :, condition_index]

# Verify the shape
print("Shape of X before reshaping:", X.shape)

X_reshaped = X.reshape(-1, 64)
print("Shape of X after reshaping:", X_reshaped.shape)

# Choose a specific channel to verify reshaping is done correctly
channel = 0

# Extract data for the specific channel
original_data = X[:, channel]
reshaped_data = X_reshaped[:, channel]

# Verify lengths for plotting
print("Length of original data for channel", channel, ":", len(original_data))
print("Length of reshaped data for channel", channel, ":", len(reshaped_data))

# Plotting
plt.figure(figsize=(14, 8))

# Plot before reshaping
plt.subplot(2, 1, 1)
plt.plot(original_data, label='Original Data')
plt.title(f'Original Data: Channel {channel}')
plt.xlabel('Flattened Time Points and Blocks')
plt.ylabel('Amplitude')
plt.legend()

# Plot after reshaping
plt.subplot(2, 1, 2)
plt.plot(reshaped_data, label='Reshaped Data')
plt.title(f'Reshaped Data: Channel {channel}')
plt.xlabel('Flattened Time Points and Blocks')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Parameters for 12 Hz
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

# Plot the first component (both reference and EEG data)
plt.figure(figsize=(10, 4))
plt.plot(X_c, label='Transformed Reference Signal (CCA Component)')
plt.plot(Y_c, label='Transformed EEG Signal (CCA Component)')
plt.title('CCA Components')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Here, I include 2 other frequencies to verify the robustness of the method
reference_frequencies = [8, 12, 16]
correlation_scores = []  # List to store correlation scores
plt.figure(figsize=(15, 10))

for freq in reference_frequencies:
    # Generate reference signals
    reference_signals = generate_reference_signals([freq], num_samples, sample_rate, num_harmonics=2)

    # Apply CCA
    cca = CCA(n_components=1)
    cca.fit(reference_signals, X_reshaped)
    X_c, Y_c = cca.transform(reference_signals, X_reshaped)

    # Compute correlation scores
    correlation_matrix = np.corrcoef(X_c.flatten(), Y_c.flatten())
    correlation_score = correlation_matrix[0, 1]
    correlation_scores.append(correlation_score)

    # Plot the CCA components
    plt.subplot(len(reference_frequencies), 1, reference_frequencies.index(freq) + 1)
    plt.plot(X_c, label='Transformed Reference Signal (CCA Component)')
    plt.plot(Y_c, label='Transformed EEG Signal (CCA Component)')
    plt.title(f'CCA Components for Reference Frequency {freq} Hz\nCorrelation Score: {correlation_score:.2f}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()