import os
import os.path
import random
from glob import glob

import numpy as np
import torch
import pypianoroll
from tqdm import tqdm

import config

## Loading data into variables
data = []
for song in tqdm(glob(config.NPZ_DIR + "/*.npz")):
    # Load the multitrack as a pypianoroll.Multitrack instance
    multitrack = pypianoroll.load(song)
    # Binarize the pianorolls
    multitrack.binarize()
    # Downsample the pianorolls (shape: n_timesteps x config.n_pitches)
    multitrack.set_resolution(config.beat_resolution)
    # Stack the pianoroll (shape: config.n_tracks x n_timesteps x config.n_pitches)
    try:
        pianoroll = multitrack.stack() > 0
    except ValueError:
        continue
    # Get the target pitch range only
    pianoroll = pianoroll[
        :, :, config.lowest_pitch : config.lowest_pitch + config.n_pitches
    ]
    if pianoroll.shape[0] != 5:
        continue
    # Calculate the total measures
    n_total_measures = multitrack.get_max_length() // config.measure_resolution
    candidate = n_total_measures - config.n_measures
    target_n_samples = min(
        n_total_measures // config.n_measures, config.n_samples_per_song
    )
    # Randomly select a number of phrases from the multitrack pianoroll
    try:
        for idx in np.random.choice(candidate, target_n_samples, False):
            start = idx * config.measure_resolution
            end = (idx + config.n_measures) * config.measure_resolution
            # Skip the samples where some track(s) has too few notes
            if (pianoroll.sum(axis=(1, 2)) < 10).any():
                continue
            data.append(pianoroll[:, start:end])
    except ValueError:
        continue
# Stack all the collected pianoroll segments into one big array
random.shuffle(data)
data = np.stack(data)
print(f"Successfully collect {len(data)} samples")
print(f"Data shape : {data.shape}")

# We create a dataset and a bootloader to the network
data = torch.as_tensor(data, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size, drop_last=True, shuffle=True
)
