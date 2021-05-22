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
    
    run_generation(generator, {})


if __name__ == "__main__":
    print("IS CUDA AVAILABLE?: ", torch.cuda.is_available())
    main()
