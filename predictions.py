import numpy as np
import torch
from pypianoroll import Multitrack, Track
from pypianoroll import Multitrack
from pypianoroll import load as midi_load
import uuid
from glob import glob

import config

from models import (
    Generator,
)


def get_file_name():
    return str(uuid.uuid4())


### Generating midi on finished models
def predict():
    generator_file_path = glob(config.ROOT_DIR + "/models/LPD/final_check_tensor*")[0]
    file_name = get_file_name()
    output_npz_filename = "%s/%s.npz" % (config.RESULTS_DIR, file_name)
    output_midi_filename = "%s/%s.mid" % (config.RESULTS_DIR, file_name)

    generator = Generator()
    sample_latent = torch.randn(config.n_samples, config.latent_dim)
    if torch.cuda.is_available():
        generator = generator.cuda()
        sample_latent = sample_latent.cuda()
    generator.load_state_dict(torch.load(generator_file_path))
    generator.eval()
    with torch.no_grad():
        samples = generator(sample_latent).cpu().detach().numpy()
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
                name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll
            )
        )
    m = Multitrack(
        tracks=tracks, tempo=config.tempo_array, resolution=config.beat_resolution
    )

    # Save .npz
    m.save(output_npz_filename)

    # Load .npz & write .mid
    m1 = midi_load(output_npz_filename)
    m1.write(output_midi_filename)
