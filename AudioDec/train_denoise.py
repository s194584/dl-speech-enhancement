# import AudioDec
import time

import numpy as np

from dataloader import CollaterAudio
from utils.audiodec import AudioDec, assign_model
from dataloading.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch
import math
import os
from os import walk

from models.autoencoder.AudioDec import Generator as generator_audiodec
import yaml
from bin.utils import load_config
from losses import DiscriminatorAdversarialLoss
from losses import FeatureMatchLoss
from losses import GeneratorAdversarialLoss
from losses import MultiMelSpectrogramLoss
from clearml import Task
import soundfile as sf
import matplotlib.pyplot as plt


ENVIRONMENT = "LAPTOP"
# ENVIRONMENT = "HPC"

if ENVIRONMENT == "LAPTOP":
    CLEAN_PATH = "corpus/train/clean"
    CLEAN_ROOT = "clean"
    NOISE_PATH = "corpus/train/noise"
    NOISE_ROOT = "noise"
    task = Task.init("dl-speech-enhancement", "Laptop")
    logger = task.get_logger()
elif ENVIRONMENT == "HPC":
    task = Task.init("dl-speech-enhancement", "HPC")
    logger = task.get_logger()
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    CLEAN_ROOT = "vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"
    NOISE_ROOT = "noise_fullband"


def add_noise(speech, noise, snr):
    if noise.shape[0] < speech.shape[0]:
        pass
    assert speech.shape == noise.shape, "Shapes are not equal!"

    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech


def trim_audio(batch):
    batch = [b for b in batch if (len(b[0]) > 1)]
    assert len(batch) > 0, f"No qualified audio pairs.!"
    return batch


def define_splits(
    dataset, generator, train_percentage=0.7, val_percentage=0.15, test_percentage=0.15
):
    split_datasets = random_split(
        dataset,
        [train_percentage, val_percentage, test_percentage],
        generator=generator,
    )
    return {
        "train": split_datasets[0],
        "validation": split_datasets[1],
        "test": split_datasets[2],
    }


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def define_optimizer_scheduler(model, config):
    generator_optimizer_class = getattr(torch.optim, config["generator_optimizer_type"])

    # discriminator_optimizer_class = getattr(
    #     torch.optim,
    #     config['discriminator_optimizer_type']
    # )

    optimizer = {
        "generator": generator_optimizer_class(
            model.parameters(),
            **config["generator_optimizer_params"],
        )
        # 'discriminator': discriminator_optimizer_class(
        #     model['discriminator'].parameters(),
        #     **config['discriminator_optimizer_params'],
        # ),
    }

    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        config.get("generator_scheduler_type", "StepLR"),
    )

    # discriminator_scheduler_class = getattr(
    #     torch.optim.lr_scheduler,
    #     config.get('discriminator_scheduler_type', "StepLR"),
    # )

    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        )
        # 'discriminator': discriminator_scheduler_class(
        #     optimizer=optimizer['discriminator'],
        #     **config['discriminator_scheduler_params'],
        # ),
    }

    return optimizer, scheduler


def define_criterion(config, device):
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(**config["generator_adv_loss_params"]).to(
            device
        ),
        "dis_adv": DiscriminatorAdversarialLoss(
            **config["discriminator_adv_loss_params"]
        ).to(device),
    }
    if config.get("use_feat_match_loss", False):
        criterion["feat_match"] = FeatureMatchLoss(
            **config.get("feat_match_loss_params", {}),
        ).to(device)

    if config.get("use_mel_loss", False):
        criterion["mel"] = MultiMelSpectrogramLoss(
            **config["mel_loss_params"],
        ).to(device)

    return criterion


# Constants for the length of noise
number_of_intervals = 2
length_of_interval = 48_000  # Number of samples in the audio file ! SHOULD BE DIVISIBLE WITH HOP_SIZE in config file


def mix_clean_noise(clean_sample, noise_sample):
    y_samples = torch.tensor([])
    x_samples = torch.tensor([])
    # for i in range(number_of_intervals):
    #     rand_clean_start = torch.randint(
    #         0, len(clean_sample) - length_of_interval, (1,)
    #     ).item()
    #     rand_noise_start = torch.randint(
    #         0, len(noise_sample) - length_of_interval, (1,)
    #     ).item()
    #
    #     clean_snippet = clean_sample[
    #         rand_clean_start : rand_clean_start + length_of_interval
    #     ]
    #     noise_snippet = noise_sample[
    #         rand_noise_start : rand_noise_start + length_of_interval
    #     ]
    #
    #     mixed_snippet = add_noise(
    #         clean_snippet, noise_snippet, 10
    #     )  # TODO - add random SNR
    #     if i == 0:
    #         y_samples = clean_snippet
    #         x_samples = mixed_snippet
    #         continue
    #     y_samples = torch.stack((y_samples, clean_snippet))
    #     x_samples = torch.stack((x_samples, mixed_snippet))

    return x_samples, y_samples


def mix_clean_noise_batch(clean_batch, noise_batch):
    y_batch = torch.tensor([])
    x_batch = torch.tensor([])
    for clean_sample in clean_batch:
        for noise_sample in noise_batch:
            x, y = mix_clean_noise(clean_sample, noise_sample)  # Maybe flatten
            x_batch = torch.cat((x_batch, x))
            y_batch = torch.cat((y_batch, y))
    return x_batch, y_batch


# Seeds for reproducibility #########
generator_seed = 81
generator = torch.Generator().manual_seed(generator_seed)

# device assignment
if ENVIRONMENT == "LAPTOP":
    tx_device = "cpu"
    rx_device = "cpu"
else:
    tx_device = "cuda:0"
    rx_device = "cuda:0"

device = torch.device(tx_device)

# Loading model #####################
# TODO - Load model
model = "vctk_denoise"
path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")
config = load_config(path_to_config)
generator_model = generator_audiodec(**config["generator_params"])
generator_model.to(device=tx_device)
optimizer, shceduler = define_optimizer_scheduler(generator_model, config)
criterion = define_criterion(config, device)
# optimizer = torch.optim.Adam(**config["generator_optim_params"])

# Freeze components
# fix quantizer
for parameter in generator_model.quantizer.parameters():
    parameter.requires_grad = False
# fix decoder
for parameter in generator_model.decoder.parameters():
    parameter.requires_grad = False

torch.set_num_threads(1)
model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)

print("AudioDec initinalizing!")
audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

# Loading data ######################
sample_rate = 48000
assert sample_rate == model_sample_rate, "The model's sample rate is not 48000Hz."
batch_length = 96_000
snr = [20, 10, 3]
clean_dataset = AudioDataset(CLEAN_PATH, CLEAN_ROOT, batch_length, sample_rate)
noise_dataset = AudioDataset(NOISE_PATH, NOISE_ROOT, batch_length, sample_rate)

clean_splits = define_splits(clean_dataset, generator)
noise_splits = define_splits(noise_dataset, generator)


collator = CollaterAudio(batch_length)

batch_size = 32
batch_size_noise = 16
train_clean_dataloader = DataLoader(
    clean_splits["train"],
    batch_size=batch_size,
    shuffle=True,
    generator=generator,
    collate_fn=collator,
)
train_noise_dataloader = DataLoader(
    noise_splits["train"],
    batch_size=batch_size_noise,
    shuffle=True,
    generator=generator,
    collate_fn=collator,
)

val_clean_dataloader = DataLoader(
    clean_splits["validation"],
    batch_size=batch_size,
    shuffle=True,
    generator=generator,
    collate_fn=collator,
)
val_noise_dataloader = DataLoader(
    noise_splits["validation"],
    batch_size=batch_size_noise,
    shuffle=True,
    generator=generator,
    collate_fn=collator,
)

# test_clean_dataloader = DataLoader(
#     clean_splits["test"],
#     batch_size=2,
#     shuffle=True,
#     generator=generator,
#     collate_fn=collator,
# )
# test_noise_dataloader = DataLoader(
#     noise_splits["test"],
#     batch_size=2,
#     shuffle=True,
#     generator=generator,
#     collate_fn=collator,
# )
# Loading Trainer ###################


def mix_and_permute_to_device(clean_sample_batch, noise_sample_batch):
    # Perform training on pseudo batch
    x_mixed, y_clean = mix_clean_noise_batch(clean_sample_batch, noise_sample_batch)
    x_mixed = x_mixed.permute(0, 2, 1).float()  # (B,C,T)
    y_clean = y_clean.permute(0, 2, 1).float()  # (B,C,T)

    # Send input through model
    x_noisy = x_mixed.to(device)
    x_target = y_clean.to(device)
    return x_noisy, x_target


def log_audio_clearml(steps):
    # y = y[0].permute(1, 0).cpu().squeeze().detach().numpy()
    # path = os.path.join("training_output", "debug.wav")
    # sf.write(
    #     path,
    #     y,
    #     sample_rate,
    #     "PCM_16",
    # )
    # logger.report_media("audio", "tada", iteration=steps, local_path=path)
    plt.plot(range(len(training_losses)), training_losses, label="Training Loss")
    plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def training_step(clean_sample_batch, mixed_sample_batch, steps):
    # x_noisy, x_target = mix_and_permute_to_device(
    #     clean_sample_batch, noise_sample_batch
    # )

    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    # fix codebook
    generator_model.quantizer.codebook.eval()  # TODO - consider if it should be put outside of this

    # initialize generator loss
    gen_loss = 0.0

    # main genertor operation
    y_nc, zq, z, vqloss, perplexity = generator_model(mixed_sample_batch)

    # perplexity info
    # self._perplexity(perplexity, mode=mode)

    # vq loss
    vqloss = torch.sum(vqloss)
    vqloss *= config["lambda_vq_loss"]
    gen_loss += vqloss

    # metric loss
    mel_loss = criterion["mel"](y_nc, clean_sample_batch)
    mel_loss *= config["lambda_mel_loss"]
    gen_loss += mel_loss

    optimizer["generator"].zero_grad()
    gen_loss.backward()

    if config["generator_grad_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(
            generator_model.parameters(),
            config["generator_grad_norm"],
        )
    optimizer["generator"].step()
    shceduler["generator"].step()

    return gen_loss


def validation_step(clean_sample_batch, mixed_sample_batch):
    # x_noisy, x_target = mix_and_permute_to_device(
    #     clean_sample_batch, noise_sample_batch
    # )

    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    # fix codebook
    generator_model.quantizer.codebook.eval()  # TODO - consider if it should be put outside of this

    # initialize generator loss
    gen_loss = 0.0

    # main genertor operation
    y_nc, zq, z, vqloss, perplexity = generator_model(mixed_sample_batch)

    # perplexity info
    # self._perplexity(perplexity, mode=mode)

    # vq loss
    vqloss = torch.sum(vqloss)
    vqloss *= config["lambda_vq_loss"]
    gen_loss += vqloss

    # metric loss
    mel_loss = criterion["mel"](y_nc, clean_sample_batch)
    mel_loss *= config["lambda_mel_loss"]
    gen_loss += mel_loss
    return gen_loss


def test():
    pass

start_time = time.perf_counter()
steps = 0
epochs = [i for i in range(3)]
training_losses = []
validation_losses = []
train_noise_iter = iter(train_noise_dataloader)
train_noise_sample_batch = next(train_noise_iter)
val_noise_iter = iter(val_noise_dataloader)
val_noise_sample_batch = next(val_noise_iter)
# Maybe have an epoch here
for epoch in epochs:
    # Training loop #####################
    i_batch = 0
    train_loss = 0
    for i_batch, clean_sample_batch in enumerate(iter(train_clean_dataloader)):
        # Sample random noise

        rand_indices = torch.randint(
            0, len(train_noise_sample_batch), (len(clean_sample_batch),)
        )
        noise_samples = train_noise_sample_batch[rand_indices]

        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_samples, 10)

        if i_batch == 3:
            break
        train_loss += training_step(clean_sample_batch, mixed_samples, steps).item()
        steps += 1


        if steps % 100 == 0:
            current_time = time.perf_counter()
            time_dif = current_time - start_time
            seconds, minutes, hours = int(time_dif % 60), int(time_dif // 60), int(time_dif // 3600)
            print(f"Training: Step {steps} \t Time: {hours}:{minutes}:{seconds}")
            # print(steps)
        # Logging in ClearML
        # if steps % 10 == 0:
        #     # print(steps)
        #     log_audio_clearml(steps)

        if 4 < torch.randint(0, 100, (1,)).item():
            try:
                train_noise_sample_batch = next(train_noise_iter)
            except:
                train_noise_dataloader = DataLoader(
                    noise_splits["train"],
                    batch_size=batch_size_noise,
                    shuffle=True,
                    generator=generator,
                    collate_fn=collator,
                )

                train_noise_iter = iter(train_noise_dataloader)
                train_noise_sample_batch = next(train_noise_iter)

    training_losses.append(train_loss / i_batch)

    # Validation loop ###################
    val_loss = 0
    i_batch = 0
    for i_batch, clean_sample_batch in enumerate(iter(val_clean_dataloader)):
        if i_batch == 3:
            break
        rand_indices = torch.randint(
            0, len(val_noise_sample_batch), (len(clean_sample_batch),)
        )
        noise_samples = val_noise_sample_batch[rand_indices]

        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_samples, 10)

        with torch.no_grad():
            val_loss += validation_step(clean_sample_batch, mixed_samples).item()


        if steps % 100 == 0:
            current_time = time.perf_counter()
            time_dif = current_time - start_time
            seconds, minutes, hours = int(time_dif % 60), int(time_dif // 60), int(time_dif // 3600)
            print(f"Validation: Step {steps} \t Time: {hours}:{minutes}:{seconds}")
        steps += 1

        # TODO - Should this be different from training
        if 10 < torch.randint(0, 100, (1,)).item():
            try:
                val_noise_sample_batch = next(val_noise_iter)
            except:
                val_noise_dataloader = val_noise_dataloader = DataLoader(
                    noise_splits["validation"],
                    batch_size=batch_size_noise,
                    shuffle=True,
                    generator=generator,
                    collate_fn=collator,
                )
                val_noise_iter = iter(val_noise_dataloader)
                val_noise_sample_batch = next(val_noise_iter)
    validation_losses.append(val_loss / i_batch)
    log_audio_clearml(steps)


plt.plot(range(len(training_losses)), training_losses, label="Training Loss")
plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# testing_losses = []
# # Testing loop ###################
# for i_batch, clean_sample_batch in enumerate(iter(test_clean_dataloader)):
#     test_loss = 0
#     # if i_batch == 3:
#     #     break
#     for j_batch, train_noise_sample_batch in enumerate(iter(test_noise_dataloader)):
#         with torch.no_grad():
#             test_loss += validation_step(clean_sample_batch, train_noise_sample_batch).item()
#     testing_losses.append(val_loss)
#
# plt.plot(testing_losses, label="Test Loss")
# plt.xlabel("Batch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
