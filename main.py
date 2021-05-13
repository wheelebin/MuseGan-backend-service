from glob import glob
import torch

from fastapi import FastAPI
from fastapi.responses import FileResponse


# Custom imports
import config
from predictions import init_generator
from utilities import make_project_dirs
from music_generation_service import run_generation


def main():
    make_project_dirs()

    generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
    generator = init_generator(generator_file_path)

    requested_operations = {
        "change_instruments": {1: 0, 2: 0, 3: 27},
        "add_drums": True,
        "add_chords": [1, 2, 3, 4, 5],
        "set_bpm": 100,
        "modify_length": 260,
        "tone_invert": True,
        "invert_midi": True,
    }

    # run_training()
    run_generation(generator, requested_operations)


if __name__ == "__main__":
    print("IS CUDA AVAILABLE?: ", torch.cuda.is_available())
    main()
