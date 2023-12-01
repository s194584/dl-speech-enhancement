# import AudioDec
import time

import numpy as np
from torch import nn
from torch.optim import Adam
from torchaudio import transforms

from dataloader import CollaterAudio
from utils.audiodec import AudioDec, assign_model
from dataloading.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics.audio import \
    (SignalNoiseRatio,
    # SignalDistortionRatio,
     ScaleInvariantSignalDistortionRatio,
    # ShortTimeObjectiveIntelligibility
     )
import torch
import math
import random
import os

from models.autoencoder.AudioDec import Generator as generator_audiodec
import yaml
from clearml import Task
import soundfile as sf
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
    task = Task.init("dl-speech-enhancement", "HPC-AudioDec-Fresh-Mel_L1-DefaultAdam")
    logger = task.get_logger()
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    CLEAN_ROOT = "vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"
    NOISE_ROOT = "noise_fullband"
else:
    raise Exception("Illegal argument: " + ENVIRONMENT)


def trim_audio(batch):
    batch = [b for b in batch if (len(b[0]) > 1)]
    assert len(batch) > 0, f"No qualified audio pairs.!"
    return batch


def add_noise(speech, noise, snr):
    assert speech.shape == noise.shape, "Shapes are not equal!"

    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech


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


# Constants for the length of noise
number_of_intervals = 2
length_of_interval = 48_000  # Number of samples in the audio file ! SHOULD BE DIVISIBLE WITH HOP_SIZE in config file

# Seeds for reproducibility #########
generator_seed = 81
random_generator = torch.manual_seed(generator_seed)

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

# Optimizer
# generator_optimizer_type: Adam
# generator_optimizer_params:
#     lr: 1.0e-4
#     betas: [0.5, 0.9]
#     weight_decay: 0.0
optimizer = Adam(generator_model.parameters())


# model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)
# TODOO - Create own config file since we don't use theirs as much anymore
def load_from_pretrained(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    generator_model.load_state_dict(state_dict["model"]["generator"])


# load_from_pretrained(encoder_checkpoint)
encoder = generator_model.encoder
encoder.to(device=tx_device)
decoder = generator_model.decoder
decoder.to(device=tx_device)

mae = nn.L1Loss().to(device)
mse = nn.MSELoss().to(device)
snr = SignalNoiseRatio().to(device)
mel_spectrogram = transforms.MelSpectrogram(48000).to(device)


def Mel_L1(pred, target):
    pred_mel = mel_spectrogram(pred)
    target_mel = mel_spectrogram(target)

    return mae(pred_mel, target_mel)


measures = {
    # 'MAE': nn.L1Loss().to(device),
    # 'MSE': nn.MSELoss().to(device),
    # 'SNR': SignalNoiseRatio().to(device),
    # 'SDR': SignalDistortionRatio().to(device),
    'SI-SDR': ScaleInvariantSignalDistortionRatio().to(device),
    # 'PESQ': PerceptualEvaluationSpeechQuality(fs=16000, mode='wb'),
    # 'STOI': ShortTimeObjectiveIntelligibility(48000),
    'Mel-L1': Mel_L1
}


def calculate_train_loss(pred, target):
    return measures['Mel-L1'](pred, target)


def calculate_validation_loss(pred, target):
    return measures['Mel-L1'](pred, target)


# Freeze components
# fix quantizer
def freeze_decoder(model):
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False


# freeze_decoder(generator_model)

# Loading data ######################
sample_rate = 48000
batch_length = 96_000
clean_dataset = AudioDataset(CLEAN_PATH, CLEAN_ROOT, batch_length, sample_rate)
noise_dataset = AudioDataset(NOISE_PATH, NOISE_ROOT, batch_length, sample_rate)

clean_splits = define_splits(clean_dataset, random_generator)
noise_splits = define_splits(noise_dataset, random_generator)

collator = CollaterAudio(batch_length)

if ENVIRONMENT == 'LAPTOP':
    batch_size = 8
    batch_size_noise = 8
else:
    batch_size = 8
    batch_size_noise = 8


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=random_generator,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        drop_last=True
    )


train_clean_dataloader = create_dataloader(clean_splits["train"], batch_size)
train_noise_dataloader = create_dataloader(noise_splits["train"], batch_size_noise)
val_clean_dataloader = create_dataloader(clean_splits["validation"], batch_size)
val_noise_dataloader = create_dataloader(noise_splits["validation"], batch_size_noise)


def log_audio_clearml(epoch, train_loss, validation_loss):
    logger.report_scalar(
        "Loss", "Train", train_loss, iteration=epoch
    )
    logger.report_scalar(
        "Loss", "Validation", validation_loss, iteration=epoch
    )


def training_step(clean_sample_batch, mixed_sample_batch):
    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    encoder.train()
    decoder.train()

    encoded_batch = encoder(mixed_sample_batch)
    y_pred = decoder(encoded_batch)

    loss = calculate_train_loss(y_pred, clean_sample_batch)

    optimizer.zero_grad()
    loss.backward()

    if config["generator_grad_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(
            generator_model.parameters(),
            config["generator_grad_norm"],
        )
    optimizer.step()

    if record_audio_snippets:
        save_snippets('prediction', y_pred.detach())

    return loss


def validation_step(clean_sample_batch, mixed_sample_batch):
    mixed_sample_batch = mixed_sample_batch.to(device)
    clean_sample_batch = clean_sample_batch.to(device)
    encoder.eval()
    decoder.eval()

    y_pred = decoder(encoder(mixed_sample_batch))

    loss = calculate_validation_loss(y_pred, clean_sample_batch)
    return loss


start_time = time.perf_counter()
steps = 0
epochs = [i for i in range(10)]
# Maybe have an epoch here
steps = 0
train_acc_batch_size = 0
train_steps = 0
record_audio_snippets = False

print("Start training")


def random_number_as_tensor(low, high, device):
    return torch.randint(low, high, (1,)).to(device)


def report_time(phase):
    current_time = time.perf_counter()
    time_dif = current_time - start_time
    seconds, minutes, hours = (
        int(time_dif % 60),
        int((time_dif // 60) % 60),
        int(time_dif // 3600),
    )
    print(f"{phase}: Step {train_steps} \t Time: {hours}:{minutes}:{seconds}")


def save_snippets(type, sample_batch):
    for i, sample in enumerate(sample_batch):
        sample = sample.squeeze(1).transpose(1, 0).cpu().numpy()
        path = os.path.join("snippets", f"{type}_snippets_{i}.wav")
        sf.write(
            path,
            sample,
            48000,
            "PCM_16",
        )


for epoch in epochs:
    # Training loop #####################
    i_batch = 0
    train_losses = []
    print("New epoch")
    for i_batch, (clean_sample_batch, noise_sample_batch) in enumerate(
            zip(train_clean_dataloader, train_noise_dataloader)):
        if len(clean_sample_batch) > len(noise_sample_batch):
            clean_sample_batch = clean_sample_batch[: len(noise_sample_batch)]
        else:
            noise_sample_batch = noise_sample_batch[: len(clean_sample_batch)]

        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_sample_batch, torch.randint(10, 20, (1,)).to(device))

        if record_audio_snippets:
            save_snippets('target', mixed_samples)

        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        loss = training_step(clean_sample_batch, mixed_samples).item()
        train_losses.append(loss)
        steps += 1
        train_steps += 1

        if steps % 100 == 0 or ENVIRONMENT == "LAPTOP":
            report_time('Training')
            logger.report_scalar(
                "Batch-loss", "Train", loss, iteration=train_steps
            )

    avg_training_loss = np.mean(train_losses)
    # Do a checkpoint
    check_point_path = os.path.join(
        "exp", "denoise", "fresh", "si-sdr", f"checkpoint-{train_steps}.pkl"
    )
    torch.save(generator_model.state_dict(), check_point_path)

    # Validation loop ###################
    val_loss = 0
    i_batch = 0
    for i_batch, (clean_sample_batch, noise_sample_batch) in enumerate(zip(val_clean_dataloader, val_noise_dataloader)):
        if len(clean_sample_batch) > len(noise_sample_batch):
            clean_sample_batch = clean_sample_batch[: len(noise_sample_batch)]
        else:
            noise_sample_batch = noise_sample_batch[: len(clean_sample_batch)]
        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        clean_sample_batch.to(device)
        noise_sample_batch.to(device)
        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_sample_batch, torch.randint(10, 20, (1,)).to(device))

        with torch.no_grad():
            val_loss += validation_step(clean_sample_batch, mixed_samples).item()

        if steps % 100 == 0 or ENVIRONMENT == "LAPTOP":
            report_time('Validation')
        steps += 1
    avg_validation_loss = val_loss / (i_batch + 1)
    log_audio_clearml(steps, avg_training_loss, avg_validation_loss)
