import os
from pathlib import Path
from tqdm import tqdm
from pypianoroll import read as midi_read
import pypianoroll
from pathlib import Path
from config import (
    CHECKPOINT_PATH,
    MIDI_DIR,
    NPZ_DIR,
    DATASET_ROOT_PATH,
    DATA_DIR,
    RESULTS_DIR,
    AMG_DIR,
)
import matplotlib.pyplot as plt


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)



def make_project_dirs():
    project_dirs = [
        CHECKPOINT_PATH,
        MIDI_DIR,
        NPZ_DIR,
        DATASET_ROOT_PATH,
        DATA_DIR,
        RESULTS_DIR,
        AMG_DIR,
    ]
    for dir in project_dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)


def convert_to_npz(input_folder, output_folder):
    for idx, song in enumerate(tqdm(Path(input_folder).glob("**/*.mid"))):
        # print(song)
        try:
            m = midi_read(song)
        except BaseException:
            continue
        # print(m)
        m.save(output_folder + f"/{idx:06}.npz")


def plot_pianoroll(file_path, file_type="npz"):
    multitrack = None
    if file_type is "npz":
        multitrack = pypianoroll.load(file_path)
    else:
        multitrack = pypianoroll.read(file_path)
    # multitrack.trim(end=12 * 96)

    axs = multitrack.plot()
    plt.gcf().set_size_inches((16, 8))
    for ax in axs:
        for x in range(96, 12 * 96, 96):
            ax.axvline(x - 0.5, color="k", linestyle="-", linewidth=1)
    plt.show()

