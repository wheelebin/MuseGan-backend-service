from glob import glob
import torch

# Custom imports
import config
from predictions import init_generator
from utilities import make_project_dirs
from music_generation_service import run_generation
from training import start_training


def main():
    make_project_dirs()

    generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
    generator = init_generator(generator_file_path)

    requested_operations = {
        "change_instruments": {"track_1": 27, "track_2": -1, "track_3": -1, "track_4": -1, "track_5": -1},
        "add_drums": True,
        "add_chords": [1, 2, 3, 4, 5],
        "set_bpm": 100,
        "modify_length": 260,
        "tone_invert": True,
        "invert_midi": True,
    }

    start_training()
    #run_generation(generator, requested_operations)


if __name__ == "__main__":
    print("IS CUDA AVAILABLE?: ", torch.cuda.is_available())
    main()
