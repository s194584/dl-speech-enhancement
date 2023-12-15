import math
import random

import numpy as np
import torch

from torch.utils.data import random_split, DataLoader

from dataloader import CollaterAudio


def add_noise(speech, noise, snr):
    assert speech.shape == noise.shape, "Shapes are not equal!"

    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech


def create_dataloader(dataset, batch_size, batch_length, generator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CollaterAudio(batch_length),
        worker_init_fn=seed_worker,
        drop_last=True,
    )


def get_dataloaders(dataset,
                    splits=None,
                    batch_size=8,
                    batch_length=2 * 48000,
                    seed=82):
    if splits is None:
        splits = [0.7, 0.15, 0.15]

    generator = torch.manual_seed(seed)
    datasets = random_split(dataset, splits, generator)
    dataloaders = []
    for fragment in datasets:
        dataloaders.append(create_dataloader(fragment, batch_size, batch_length, generator))
    return dataloaders


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
