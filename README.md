# ssvep_cca
Python code explaining why Canonical Correlation Analysis (CCA) works in detecting Steady State Visually Evoked Potentials (SSVEP).

# Data
The data used in this repository is freely available [here](https://springernature.figshare.com/collections/An_open_dataset_for_human_SSVEPs_in_the_frequency_range_of_1-60_Hz/6752910/1)

# Example

I used the dataset `data_s19_64.mat` and set the block ID to 2; frequency = 12 Hz; condition = low depth. The plot shows the comparison when using reference signals of 8 Hz, 12 Hz, and 16 Hz. As can be seen, despite the attempts to find linear combinations of a multi-channel EEG data for each of the reference signals, the target frequency (12 Hz) achieves the highest canonical correlation.

![Fig03_3frequencies](https://github.com/user-attachments/assets/27583d2a-db4c-4e46-98e3-709ec9c4e387)
