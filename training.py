from IPython.display import clear_output
import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from pypianoroll import Multitrack, Track
import numpy as np
import matplotlib.pyplot as plt


import training
import config
from models import (
    Generator,
    Discriminator,
)
from dataLoaders.LPD import get_lpd_dataloader

### Functions for training the network
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute the gradient penalty for regularization. Intuitively, the
    gradient penalty help stablize the magnitude of the gradients that the
    discriminator provides to the generator, and thus help stablize the training
    of the generator."""
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)
    # Get gradients w.r.t. the interpolations
    fake = torch.ones(real_samples.size(0), 1).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_one_step(discriminator, generator, d_optimizer, g_optimizer, real_samples):
    """Train the networks for one step."""
    # Sample from the lantent distribution
    latent = torch.randn(config.batch_size, config.latent_dim)

    # Transfer data to GPU
    if torch.cuda.is_available():
        real_samples = real_samples.cuda()
        latent = latent.cuda()

    # === Train the discriminator ===
    # Reset cached gradients to zero
    d_optimizer.zero_grad()
    # Get discriminator outputs for the real samples
    prediction_real = discriminator(real_samples)
    # Compute the loss function
    # d_loss_real = torch.mean(torch.nn.functional.relu(1. - prediction_real))
    d_loss_real = -torch.mean(prediction_real)
    # Backpropagate the gradients
    d_loss_real.backward()

    # Generate fake samples with the generator
    fake_samples = generator(latent)
    # Get discriminator outputs for the fake samples
    prediction_fake_d = discriminator(fake_samples.detach())
    # Compute the loss function
    # d_loss_fake = torch.mean(torch.nn.functional.relu(1. + prediction_fake_d))
    d_loss_fake = torch.mean(prediction_fake_d)
    # Backpropagate the gradients
    d_loss_fake.backward()

    # Compute gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(
        discriminator, real_samples.data, fake_samples.data
    )
    # Backpropagate the gradients
    gradient_penalty.backward()

    # Update the weights
    d_optimizer.step()

    # === Train the generator ===
    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # Get discriminator outputs for the fake samples
    prediction_fake_g = discriminator(fake_samples)
    # Compute the loss function
    g_loss = -torch.mean(prediction_fake_g)
    # Backpropagate the gradients
    g_loss.backward()
    # Update the weights
    g_optimizer.step()

    return d_loss_real + d_loss_fake, g_loss


def start_training():
    # Prepare data
    print("Loading LPD data")
    lpd_dataset_path = config.DATA_DIR + "/lp_5_clensed_tensor_dataset.pt"
    data_loader = get_lpd_dataloader(lpd_dataset_path)

    ### Preparing to train the network
    # Create neural networks
    print("Setting up discriminator")
    discriminator = Discriminator()
    print("Setting up generator")
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
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.001, betas=(0.5, 0.9)
    )
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
    print("Start training")
    while step < config.n_steps + 1:
        # Iterate over the dataset
        for real_samples in data_loader:
            # Train the neural networks
            generator.train()
            d_loss, g_loss = training.train_one_step(
                discriminator, generator, d_optimizer, g_optimizer, real_samples[0]
            )
            # Save pytorch model
            if d_loss < d_loss_min:
                torch.save(
                    generator.state_dict(),
                    # config.CHECKPOINT_PATH + str(step) + "_check_" + str(d_loss),
                    "%s/%s_check_%s" % (config.CHECKPOINT_PATH, str(step), str(d_loss))
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

            # Code in if statmnent is for generating samples while training &
            # for plotting out piano roll's
            # I might as well remove this from the code
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
                            (
                                config.lowest_pitch,
                                128 - config.lowest_pitch - config.n_pitches,
                            ),
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
                m = Multitrack(
                    tracks=tracks,
                    tempo=config.tempo_array,
                    resolution=config.beat_resolution,
                )
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

    print("Attempting to post training save")
    torch.save(
        generator.state_dict(),
        config.CHECKPOINT_PATH + "/final_check_" + str(d_loss),
    )
    print("Post training save susccesful!")
