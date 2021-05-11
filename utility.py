from pathlib import Path
from tqdm import tqdm
from pypianoroll import read as midi_read


def convert_to_npz(input_folder, output_folder):
    """I feel like this is more of a utility function"""
    for idx, song in enumerate(tqdm(Path(input_folder).glob("**/*.mid"))):
        # print(song)
        try:
            m = midi_read(song)
        except BaseException:
            continue
        # print(m)
        m.save(output_folder + f"/{idx:06}.npz")
