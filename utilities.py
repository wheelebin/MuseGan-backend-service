from pathlib import Path
from tqdm import tqdm
from pypianoroll import read as midi_read
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
