from glob import glob
from mido.midifiles.tracks import fix_end_of_track
import torch

from fastapi import FastAPI
from fastapi.responses import FileResponse

import music21 as m21

# Custom imports
import config
from predictions import predict, init_generator
from utilities import make_project_dirs, plot_pianoroll
from midi_utilities import (
    convert_midi_to_wav,
    notes_to_chords,
    change_instruments_to_piano,
)

#%%
from utilities import make_project_dirs, plot_pianoroll


# print("Plot pianoroll")
# plot_pianoroll("data/results/c9939344-085c-4a3d-bec5-ba02cdaeb9be.mid", "mid")
# print("Finished plotting pianoroll")
#%%

# Todo
# Organize project code into a flat structure inside of src
# Figure out good way to organise data needed and generated by this project
def run_generation(generator):
    # Turn this into its own service file, called music generator or something
    file_name, output_midi_filename, output_npz_filename = predict(generator)
    print(file_name, output_midi_filename, output_npz_filename)

    # TODO this is not working so make check to see why this wont create a playable wav and if any other methods are messing up
    notes_to_chords(output_midi_filename, file_name + "_chords.mid")


    # change_instruments_to_piano(
    #     output_midi_filename,
    #     file_name + "_pianos.mid",
    #     {1: 76, 2: 76, 3: 76, 4: 76, 5: 76},
    # )

    convert_midi_to_wav(
        output_midi_filename, file_name + ".wav"
    )  # , config.SOUNDFONTS_DIR + "/kit3.sf2"

    convert_midi_to_wav(
        file_name + "_chords.mid", file_name + "_chords.wav"
    )  # , config.SOUNDFONTS_DIR + "/kit3.sf2"


def main():
    make_project_dirs()

    generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
    generator = init_generator(generator_file_path)

    # run_training()
    run_generation(generator)


if __name__ == "__main__":
    print("IS CUDA AVAILABLE?: ", torch.cuda.is_available())
    main()
