
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


CHECKPOINT_PATH = ROOT_DIR + "/models/LPD"
MIDI_DIR = ROOT_DIR + "/midi_train/Classical/Bach"
NPZ_DIR = ROOT_DIR + "/npz_train/Bach"
DATASET_ROOT_PATH = ROOT_DIR + "/data/lpd_5/lpd_5_cleansed"
DATA_DIR = ROOT_DIR + "/data"
RESULTS_DIR = DATA_DIR + "/results"
AMG_DIR = DATA_DIR + "/amg"

# Data
n_tracks = 5  # number of tracks
n_pitches = 72  # number of pitches
lowest_pitch = 24  # MIDI note number of the lowest pitch
n_samples_per_song = 8  # number of samples to extract from each song in the datset
n_measures = 4  # number of measures per sample
beat_resolution = 4  # temporal resolution of a beat (in timestep)
programs = [0, 0, 25, 33, 48]  # program number for each track
is_drums = [True, False, False, False, False]  # drum indicator for each track
track_names = ["Drums", "Piano", "Guitar", "Bass", "Strings"]  # name of each track
tempo = 100
# Training
batch_size = 16
latent_dim = 128
n_steps = 20000
# Sampling
sample_interval = 100  # interval to run the sampler (in step)
n_samples = 4

measure_resolution = 4 * beat_resolution
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)

""" assert 24 % beat_resolution == 0, (
    "beat_resolution must be a factor of 24 (the beat resolution used in "
    "the source dataset)."
)
assert len(programs) == len(is_drums) and len(programs) == len(
    track_names
), "Lengths of programs, is_drums and track_names must be the same."
 """
