# import AudioDec
import time

import numpy as np

from dataloader import CollaterAudio
from utils.audiodec import AudioDec, assign_model
from dataloading.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch
import math
import random
import os

from models.autoencoder.AudioDec import Generator as generator_audiodec
import yaml
from bin.utils import load_config
from losses import DiscriminatorAdversarialLoss
from losses import FeatureMatchLoss
from losses import GeneratorAdversarialLoss
from losses import MultiMelSpectrogramLoss
from clearml import Task, Logger
import soundfile as sf
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--environment", default="LAPTOP")

args = parser.parse_args()


ENVIRONMENT = args.environment
# ENVIRONMENT = "HPC"

if ENVIRONMENT == "LAPTOP":
    CLEAN_PATH = "corpus/train/clean"
    CLEAN_ROOT = "clean"
    NOISE_PATH = "corpus/train/noise"
    NOISE_ROOT = "noise"
    task = Task.init("dl-speech-enhancement", "Laptop")
    logger = task.get_logger()
    torch.set_num_threads(4)
elif ENVIRONMENT == "HPC":
    task = Task.init("dl-speech-enhancement", "HPC-AudioDec")
    logger = task.get_logger()
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    CLEAN_ROOT = "vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"
    NOISE_ROOT = "noise_fullband"
else:
    raise Exception("Illegal argument: " + ENVIRONMENT)


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

# Seeds for reproducibility #########
generator_seed = 81
random_generator = torch.manual_seed(generator_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# device assignment
if ENVIRONMENT == "LAPTOP":
    tx_device = "cpu"
    rx_device = "cpu"
else:
    tx_device = "cuda:0"
    rx_device = "cuda:0"

device = torch.device(tx_device)

# Loading model #####################
model = "vctk_denoise"
path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")

config = load_config(path_to_config)
generator_model = generator_audiodec(**config["generator_params"])
generator_model.to(device=tx_device)
optimizer, scheduler = define_optimizer_scheduler(generator_model, config)
criterion = define_criterion(config, device)

model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)

state_dict = torch.load(encoder_checkpoint, map_location="cpu")
generator_model.load_state_dict(state_dict["model"]["generator"])


# EVAL
def reconstruct_test(model):
    clean_file_path = "corpus/test/book_00002_chp_0005_reader_11980_3_seg_0.wav"
    noise_file_path = "corpus/train/noise/_1eSheWjfJQ.wav"
    pred_output_path = f"job_out/book_00002_chp_0005_reader_11980_3_seg_0___{model}___reconstructed.wav"
    mixed_output_path = (
        f"job_out/book_00002_chp_0005_reader_11980_3_seg_0___{model}___mixed.wav"
    )
    clean_audio = np.array(sf.read(clean_file_path)[0])
    noise_audio = np.array(sf.read(noise_file_path)[0])

    if len(clean_audio) > len(noise_audio):
        clean_audio = clean_audio[: len(noise_audio)]
    else:
        noise_audio = noise_audio[: len(clean_audio)]

    audio = add_noise(torch.tensor(clean_audio), torch.tensor(noise_audio), 10)
    audio = audio.numpy()

    sf.write(
        mixed_output_path,
        audio,
        48000,
        "PCM_16",
    )

    audio = np.expand_dims(audio, axis=0)  # (T, C) -> (C, 1, T)
    audio = np.expand_dims(audio, axis=0)  # (T, C) -> (C, 1, T)
    with torch.no_grad():
        audio = torch.tensor(audio, dtype=torch.float).to(device)
        y = generator_model(audio)[0].squeeze(1).transpose(1, 0).cpu().numpy()
        sf.write(
            pred_output_path,
            y,
            48000,
            "PCM_16",
        )


if ENVIRONMENT == "LAPTOP":
    reconstruct_test("generator")


# Freeze components
# fix quantizer
for parameter in generator_model.quantizer.parameters():
    parameter.requires_grad = False
# fix decoder
for parameter in generator_model.decoder.parameters():
    parameter.requires_grad = False

# Loading data ######################
sample_rate = 48000
batch_length = 96_000
snr = [20, 10, 3]
clean_dataset = AudioDataset(CLEAN_PATH, CLEAN_ROOT, batch_length, sample_rate)
noise_dataset = AudioDataset(NOISE_PATH, NOISE_ROOT, batch_length, sample_rate)

clean_splits = define_splits(clean_dataset, random_generator)
noise_splits = define_splits(noise_dataset, random_generator)

collator = CollaterAudio(batch_length)

batch_size = 16
batch_size_noise = 8
train_clean_dataloader = DataLoader(
    clean_splits["train"],
    batch_size=batch_size,
    shuffle=True,
    generator=random_generator,
    collate_fn=collator,
    worker_init_fn=seed_worker,
)
train_noise_dataloader = DataLoader(
    noise_splits["train"],
    batch_size=batch_size_noise,
    shuffle=True,
    generator=random_generator,
    collate_fn=collator,
    worker_init_fn=seed_worker,
)

val_clean_dataloader = DataLoader(
    clean_splits["validation"],
    batch_size=batch_size,
    shuffle=True,
    generator=random_generator,
    collate_fn=collator,
    worker_init_fn=seed_worker,
)
val_noise_dataloader = DataLoader(
    noise_splits["validation"],
    batch_size=batch_size_noise,
    shuffle=True,
    generator=random_generator,
    collate_fn=collator,
    worker_init_fn=seed_worker,
)

# Loading Trainer ###################


def log_audio_clearml(steps):
    # TODO - Report scalar would maybe speed this process up
    plt.plot(range(len(training_losses)), training_losses, label="Training Loss")
    plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def training_step(clean_sample_batch, mixed_sample_batch):
    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    # fix codebook
    generator_model.quantizer.codebook.eval()

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
    scheduler["generator"].step()

    return gen_loss


def validation_step(clean_sample_batch, mixed_sample_batch):
    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    # fix codebook
    generator_model.quantizer.codebook.eval()

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
epochs = [i for i in range(5)]
training_losses = []
validation_losses = []
train_noise_iter = iter(train_noise_dataloader)
train_noise_sample_batch = next(train_noise_iter)
val_noise_iter = iter(val_noise_dataloader)
val_noise_sample_batch = next(val_noise_iter)


def load_next_noise(noise_iterator, dataset_name):
    try:
        next_noise_batch = next(noise_iterator)
    except:
        noise_dataloader = DataLoader(
            noise_splits[dataset_name],
            batch_size=batch_size_noise,
            shuffle=True,
            generator=random_generator,
            worker_init_fn=seed_worker,
            collate_fn=collator,
        )

        noise_iterator = iter(noise_dataloader)
        next_noise_batch = next(noise_iterator)

    return next_noise_batch, noise_iterator


# Maybe have an epoch here
steps = 0
train_acc_batch_size = 0
train_steps = 0

print("Start training")

for epoch in epochs:
    # Training loop #####################
    i_batch = 0
    train_loss = 0
    train_losses = []
    for i_batch, clean_sample_batch in enumerate(iter(train_clean_dataloader)):
        # Sample random noise

        rand_indices = torch.randint(
            0, len(train_noise_sample_batch), (len(clean_sample_batch),)
        )
        noise_samples = train_noise_sample_batch[rand_indices]

        # clean_sample_batch.to(device)
        # noise_samples.to(device)

        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_samples, 10)

        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        loss = training_step(clean_sample_batch, mixed_samples).item()
        train_loss += loss
        train_losses.append(loss)
        steps += 1
        train_steps += 1

        if steps % 100 == 0:
            current_time = time.perf_counter()
            time_dif = current_time - start_time
            seconds, minutes, hours = (
                int(time_dif % 60),
                int(time_dif // 60),
                int(time_dif // 3600),
            )
            print(f"Training: Step {train_steps} \t Time: {hours}:{minutes}:{seconds}")
            # TODO - Report scalar
            if ENVIRONMENT == "HPC":
                for loss_ in training_losses:
                    logger.report_scalar(
                        "Train-loss", "Train", loss_, iteration=train_steps
                    )
                training_losses = []

        if 4 < torch.randint(0, 100, (1,)).item():
            next_noise_batch, train_noise_iter = load_next_noise(
                train_noise_iter, "train"
            )

    training_losses.append(train_loss / (i_batch + 1))
    # Do a checkpoint
    check_point_path = os.path.join(
        "exp", "denoise", "from_pretrained", f"checkpoint-{train_steps}.pkl"
    )
    torch.save(generator_model.state_dict(), check_point_path)
    reconstruct_test(f"AudioDec-{train_steps}")

    # Validation loop ###################
    val_loss = 0
    i_batch = 0
    for i_batch, clean_sample_batch in enumerate(iter(val_clean_dataloader)):
        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        rand_indices = torch.randint(
            0, len(val_noise_sample_batch), (len(clean_sample_batch),)
        )
        noise_samples = val_noise_sample_batch[rand_indices]

        clean_sample_batch.to(device)
        noise_samples.to(device)
        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_samples, 10)

        with torch.no_grad():
            val_loss += validation_step(clean_sample_batch, mixed_samples).item()

        if steps % 100 == 0:
            current_time = time.perf_counter()
            time_dif = current_time - start_time
            seconds, minutes, hours = (
                int(time_dif % 60),
                int(time_dif // 60),
                int(time_dif // 3600),
            )
            print(f"Validation: Step {steps} \t Time: {hours}:{minutes}:{seconds}")
        steps += 1

        # TODO - Should this be different from training
        if 4 < torch.randint(0, 100, (1,)).item():
            next_noise_batch, val_noise_iter = load_next_noise(
                val_noise_iter, "validation"
            )

    validation_losses.append(val_loss / (i_batch + 1))
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
