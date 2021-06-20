from glob import glob
import torch

# Custom imports
import config
from utilities import make_project_dirs
from music_generation_service import MusicGenKing
from training import start_training


def main():
    make_project_dirs()

    print("Cuda is available: ", torch.cuda.is_available())

    generator_file_path = glob(config.CHECKPOINT_PATH + "/tensor_*")[0]
    genKing = MusicGenKing(generator_file_path)

    while True:
        print("What program do you want to run?")
        program = "p"  # input('["train" or "t" / "predict" or "p"] ')

        if program == "train" or program == "t":
            start_training()
            print("Finished training")
            break

        elif program == "predict" or program == "p":

            gen_type = "ai2"  # input('Choose a generator type ["ai1" / "ai2"] ')

            if gen_type == "ai1" or gen_type == "ai2":

                requested_operations = {
                    "change_instruments": {
                        "track_1": 27,
                        "track_2": -1,
                        "track_3": -1,
                        "track_4": -1,
                        "track_5": -1,
                    },
                    "add_drums": True,
                    "add_chords": [1, 2, 3, 4, 5],
                    "set_bpm": 100,
                    "modify_length": 260,
                    "tone_invert": True,
                    "invert_midi": True,
                }

                genKing.run_generation(gen_type, requested_operations)
                break

        print('\nThe program "%s" was not recognized!\n' % program)


if __name__ == "__main__":

    print("[Cuda is availability is: ", torch.cuda.is_available(), "]\n")
    main()
