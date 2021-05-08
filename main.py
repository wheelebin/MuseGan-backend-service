from IPython.display import clear_output
from ipywidgets import interact, IntSlider

import os
import os.path
import random
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

from pypianoroll import Multitrack
from pypianoroll import load as midi_load
from pypianoroll import read as midi_read

# Custom imports
import training
import config

from models import (
    GeneraterBlock,
    Generator,
    LayerNorm,
    DiscriminatorBlock,
    Discriminator,
)


### Preparing to train the network
# Create neural networks
discriminator = Discriminator()
generator = Generator()
print(
    "Number of parameters in G: {}".format(
        sum(p.numel() for p in generator.parameters() if p.requires_grad)
    )
)
print(
    "Number of parameters in D: {}".format(
        sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    )
)

# Create optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.9))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.9))

# Prepare the inputs for the sampler, which wil run during the training
sample_latent = torch.randn(config.n_samples, config.latent_dim)

# Transfer the neural nets and samples to GPU
if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    sample_latent = sample_latent.cuda()

# Create an empty dictionary to sotre history samples
history_samples = {}

# Create a LiveLoss logger instance for monitoring
liveloss = PlotLosses(outputs=[MatplotlibPlot(cell_size=(6, 2))])

# Initialize step
step = 0


### Train and visualize the loss function and resulting examples
# Create a progress bar instance for monitoring
progress_bar = tqdm(total=config.n_steps, initial=step, ncols=80, mininterval=1)
d_loss_min = 1000
# Start iterations
while step < config.n_steps + 1:
    # Iterate over the dataset
    for real_samples in data_loader:
        # Train the neural networks
        generator.train()
        d_loss, g_loss = training.train_one_step(
            d_optimizer, g_optimizer, real_samples[0]
        )
        if d_loss < d_loss_min:
            torch.save(
                generator.state_dict(),
                config.CHECKPOINT_PATH + str(step) + "_check_" + str(d_loss),
            )
            d_loss_min = d_loss
        # Record smoothened loss values to LiveLoss logger
        if step > 0:
            running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
            running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
        else:
            running_d_loss, running_g_loss = 0.0, 0.0
        liveloss.update({"negative_critic_loss": -running_d_loss})
        # liveloss.update({'d_loss': running_d_loss, 'g_loss': running_g_loss})

        # Update losses to progress bar
        progress_bar.set_description_str(
            "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss)
        )

        if step % config.sample_interval == 0:
            # Get generated samples
            generator.eval()
            samples = generator(sample_latent).cpu().detach().numpy()
            history_samples[step] = samples

            # Display loss curves
            clear_output(True)
            if step > 0:
                liveloss.send()

            # Display generated samples
            samples = samples.transpose(1, 0, 2, 3).reshape(config.n_tracks, -1, config.n_pitches)
            tracks = []
            for idx, (program, is_drum, track_name) in enumerate(
                zip(config.programs, config.is_drums, config.track_names)
            ):
                pianoroll = np.pad(
                    samples[idx] > 0.5,
                    ((0, 0), (config.lowest_pitch, 128 - config.lowest_pitch - config.n_pitches)),
                )
                tracks.append(
                    Track(
                        name=track_name,
                        program=program,
                        is_drum=is_drum,
                        pianoroll=pianoroll,
                    )
                )
            m = Multitrack(tracks=tracks, tempo=config.tempo_array, resolution=config.beat_resolution)
            axs = m.plot()
            plt.gcf().set_size_inches((16, 8))
            for ax in axs:
                for x in range(
                    config.measure_resolution,
                    4 * config.measure_resolution * config.n_measures,
                    config.measure_resolution,
                ):
                    if x % (config.measure_resolution * 4) == 0:
                        ax.axvline(x - 0.5, color="k")
                    else:
                        ax.axvline(x - 0.5, color="k", linestyle="-", linewidth=1)
            plt.show()

        step += 1
        progress_bar.update(1)
        if step >= config.n_steps:
            break


### Generating midi on finished models
def generate_midi(generator_parameters, output_filename):
    generator = Generator()
    sample_latent = torch.randn(config.n_samples, config.latent_dim)
    if torch.cuda.is_available():
        generator = generator.cuda()
        sample_latent = sample_latent.cuda()
    generator.load_state_dict(torch.load(generator_parameters))
    generator.eval()
    with torch.no_grad():
        samples = generator(sample_latent).cpu().detach().numpy()
    samples = samples.transpose(1, 0, 2, 3).reshape(config.n_tracks, -1, config.n_pitches)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config.programs, config.is_drums, config.track_names)
    ):
        pianoroll = np.pad(
            samples[idx] > 0.5, ((0, 0), (config.lowest_pitch, 128 - config.lowest_pitch - config.n_pitches))
        )
        tracks.append(
            Track(
                name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll
            )
        )
    m = Multitrack(tracks=tracks, tempo=config.tempo_array, resolution=config.beat_resolution)
    m.save("out.npz")
    m1 = midi_load("out.npz")
    m1.write(output_filename)
