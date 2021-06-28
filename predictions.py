import numpy as np
import torch

# from pypianoroll import Multitrack, Track
# from pypianoroll import Multitrack
# from pypianoroll import load as midi_load

from utils.musegan.src.musegan.packages.pypianoroll import Multitrack, Track
from utils.musegan.src.musegan.packages.pypianoroll import load as midi_load


import uuid
from glob import glob
from utils.musegan.src.inference import Inference

import config
from utilities import get_file_name_for_saving

from models import (
    Generator,
)

from midi_utilities import tonal_inversion, invert_midi, get_midi_by_genre

class GenOne:
    def __init__(self, generator_file_path):

        # NOTE: Maybe this should live somewhere else

        self.generator = Generator()
        map_location = None

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
        else:
            map_location = torch.device("cpu")

        self.generator.load_state_dict(torch.load(generator_file_path, map_location))
        self.generator.eval()  # NOTE: I might consider moving this and forcing you to evail the model yourself

    def predict(self, file_name, _tempo=None):
        file_name, output_npz_filename, *_ = get_file_name_for_saving("npz", file_name)
        file_name, output_midi_filename, *_ = get_file_name_for_saving("mid", file_name)

        sample_latent = torch.randn(config.n_samples, config.latent_dim)
        if torch.cuda.is_available():
            sample_latent = sample_latent.cuda()

        with torch.no_grad():
            samples = self.generator(sample_latent).cpu().detach().numpy()
        samples = samples.transpose(1, 0, 2, 3).reshape(
            config.n_tracks, -1, config.n_pitches
        )
        tracks = []
        for idx, (program, is_drum, track_name) in enumerate(
            zip(config.programs, config.is_drums, config.track_names)
        ):
            pianoroll = np.pad(
                samples[idx] > 0.5,
                (
                    (0, 0),
                    (config.lowest_pitch, 128 - config.lowest_pitch - config.n_pitches),
                ),
            )
            tracks.append(
                Track(
                    name=track_name,
                    program=program,
                    is_drum=is_drum,
                    pianoroll=pianoroll,
                )
            )

        # TODO: DO THIS SOMEWHER EELSE OR GET VALUES FROM CONFIG
        tempo_array = config.tempo_array
        
        if _tempo is not None:
            beat_resolution = 4
            measure_resolution = 4 * beat_resolution
            tempo_array = np.full((4 * 4 * measure_resolution, 1), _tempo)

        m = Multitrack(
            tracks=tracks, tempo=tempo_array, resolution=config.beat_resolution
        )

        print("GEN ONE OUTPUT")
        print(output_npz_filename)
        print(output_midi_filename)

        # Save .npz
        m.save(output_npz_filename)

        # Load .npz & write .mid
        m1 = midi_load(output_npz_filename)
        m1.write(output_midi_filename)

        return (file_name, output_midi_filename, output_npz_filename)


class GenTwo:
    def __init__(self):
        self.inference = Inference()

    def predict(self, file_name, tempo=None):
        result = self.inference.predict()

        file_name, output_npz_filename, *_ = get_file_name_for_saving("npz", file_name)
        file_name, output_midi_filename, *_ = get_file_name_for_saving("mid", file_name)

        print("GEN TWO OUTPUT")
        print(result)
        print(output_npz_filename)
        print(output_midi_filename)

        # Save .npz
        self.inference.save_pianoroll(result, output_npz_filename, tempo)

        # Load .npz & write .mid
        m1 = midi_load(output_npz_filename)
        m1.write(output_midi_filename)

        return (file_name, output_midi_filename, output_npz_filename)

class GenThree:
    def __init__(self):
        pass

    def predict(self, file_name, genre):

        midi = get_midi_by_genre(genre)
        if midi == None:
            return None

        file_name, output_midi_filename, *_ = get_file_name_for_saving("mid", file_name)
        
        # Bellow might be running unfiltered once so would have to be a loop with a try except and break on sucess result
        result = tonal_inversion(midi)

        result.save(output_midi_filename)

        return (file_name, output_midi_filename)







class GenFour:
    def __init__(self):
        pass

    def predict(self, file_name, genre):

        midi = get_midi_by_genre(genre)
        if midi == None:
            return None

        file_name, output_midi_filename, *_ = get_file_name_for_saving("mid", file_name)
        
        # Bellow might be running unfiltered once so would have to be a loop with a try except and break on sucess result
        result = invert_midi(midi)

        result.save(output_midi_filename)

        return (file_name, output_midi_filename)