import os
import os.path
import random
from pathlib import Path
from glob import glob

import numpy as np
import torch
import pypianoroll
from tqdm import tqdm

import config

from utilities import msd_id_to_dirs


def get_lpd_dataloader(pt_file_path=""):
    dataset = None
    if pt_file_path is "":
        print("CREATE TENSOR DATASET")
        """Prepairing data based on LPD"""
        dataset_root = Path(config.DATASET_ROOT_PATH)

        id_list = []
        for path in os.listdir(config.AMG_DIR):
            filepath = os.path.join(config.AMG_DIR, path)
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    id_list.extend([line.rstrip() for line in f])
        id_list = list(set(id_list))

        n = 0

        ## Loading data into variables
        data = []
        # Iterate over all the songs in the ID list
        for msd_id in tqdm(id_list):
            if n > 100:
                break
            n = n + 1
            print(n)
            # Load the multitrack as a pypianoroll.Multitrack instance
            song_dir = dataset_root / msd_id_to_dirs(msd_id)
            multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
            # Binarize the pianorolls
            multitrack.binarize()
            # Downsample the pianorolls (shape: n_timesteps x config.n_pitches)
            multitrack.set_resolution(config.beat_resolution)
            # Stack the pianoroll (shape: config.n_tracks x n_timesteps x config.n_pitches)
            pianoroll = multitrack.stack() > 0
            # Get the target pitch range only
            pianoroll = pianoroll[
                :, :, config.lowest_pitch : config.lowest_pitch + config.n_pitches
            ]
            # Calculate the total measures
            n_total_measures = multitrack.get_max_length() // config.measure_resolution
            candidate = n_total_measures - config.n_measures
            target_n_samples = min(
                n_total_measures // config.n_measures, config.n_samples_per_song
            )
            # Randomly select a number of phrases from the multitrack pianoroll
            for idx in np.random.choice(candidate, target_n_samples, False):
                start = idx * config.measure_resolution
                end = (idx + config.n_measures) * config.measure_resolution
                # Skip the samples where some track(s) has too few notes
                if (pianoroll.sum(axis=(1, 2)) < 10).any():
                    continue
                data.append(pianoroll[:, start:end])

        print(data)

        # Stack all the collected pianoroll segments into one big array
        random.shuffle(data)
        data = np.stack(data)
        print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
        print(f"Data shape : {data.shape}")

        ## We create a dataset and a data loader to the network
        data = torch.as_tensor(data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(data)
        torch.save(dataset, "lp_5_clensed_tensor_dataset.pt")
    else:
        print("LOAD TENSOR DATASET: ", pt_file_path)

        map_location = None
        if (
            torch.cuda.is_available() is False
        ):  # TODO Move the cuda.is_available() calls to the config file and have it set once on startup
            map_location = torch.device("cpu")

        dataset = torch.load(pt_file_path, map_location)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return data_loader
