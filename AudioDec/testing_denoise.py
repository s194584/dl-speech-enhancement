# import AudioDec

import numpy as np
import torchaudio

from dataloader import CollaterAudio
from dataloader.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch
import math
import random
import os

from models.autoencoder_without_PQC.AudioDec import Generator as generator_audiodec
from models.autoencoder.AudioDec import Generator as generator_audiodec_original
import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--environment", default="LAPTOP")
parser.add_argument("-sr", "--sample_rate", default=48000)

args = parser.parse_args()
SAMPLE_RATE = int(args.sample_rate)
ENVIRONMENT = args.environment

if ENVIRONMENT == "LAPTOP":
    CLEAN_PATH = "corpus/train/clean"
    CLEAN_ROOT = "clean"
    NOISE_PATH = "corpus/train/noise"
    NOISE_ROOT = "noise"
    torch.set_num_threads(4)
elif ENVIRONMENT == "HPC":
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
def define_AD_model():
    # path_to_model_dir = os.path.join("exp","denoise","symAD_vctk_48000_hop300","config.yml")
    path_to_config = os.path.join("exp","denoise","symAD_vctk_48000_hop300","config.yml")
    path_to_model = os.path.join("exp","denoise","symAD_vctk_48000_hop300","checkpoint-200000steps.pkl")
    config = load_config(path_to_config)
    generator = generator_audiodec_original(**config['generator_params'])
    state_dict = torch.load(path_to_model)
    generator.load_state_dict(state_dict['model']['generator'])
    generator = generator.eval().to(device)
    return generator

def load_flagship():
    path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")
    path_to_model = os.path.join("exp","denoise","HPC-AudioDec-Fresh-Mel_L1-MAE-Advcheckpoint-22332.pkl")
    config = load_config(path_to_config)
    generator = generator_audiodec(**config['generator_params'])
    state_dict = torch.load(path_to_model, map_location=torch.device('cpu'))
    generator.load_state_dict(state_dict)
    return generator


# Loading data ######################
batch_length = 2 * SAMPLE_RATE
clean_dataset = AudioDataset(CLEAN_PATH, CLEAN_ROOT, batch_length, SAMPLE_RATE)
noise_dataset = AudioDataset(NOISE_PATH, NOISE_ROOT, batch_length, SAMPLE_RATE)

clean_splits = define_splits(clean_dataset, random_generator)
noise_splits = define_splits(noise_dataset, random_generator)

collator = CollaterAudio(batch_length)

if ENVIRONMENT == 'LAPTOP':
    batch_size = 5
else:
    batch_size = 8


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
        worker_init_fn=seed_worker
    )

train_clean_dataloader = create_dataloader(clean_splits["train"], batch_size)
train_noise_dataloader = create_dataloader(noise_splits["train"], batch_size)
test_clean_dataloader = create_dataloader(clean_splits["test"], batch_size)
test_noise_dataloader = create_dataloader(noise_splits["test"], batch_size)


SAMPLE_RATE = 48000
models = {"AudioDec_Denoise": define_AD_model(), "Ours": load_flagship()}

# make test directories
observation_counters = {}
for model_name in models:
    path = os.path.join("test_out",model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    observation_counters[model_name] = 0

def infer(clean_dataloader,noise_dataloader):
    for i_batch, (clean_sample_batch, noise_sample_batch) in enumerate(
            zip(clean_dataloader, noise_dataloader)):

        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        if len(clean_sample_batch) > len(noise_sample_batch):
            clean_sample_batch = clean_sample_batch[: len(noise_sample_batch)]
        else:
            noise_sample_batch = noise_sample_batch[: len(clean_sample_batch)]

        # Mix noise
        mixed_samples = add_noise(clean_sample_batch, noise_sample_batch, torch.randint(10, 20, (1,)).to(device))

        for model_name, model in models.items():
            with torch.no_grad():
                y = model(mixed_samples)
                if isinstance(y,tuple) or isinstance(y,list):
                    y = y[0]
                y = y.detach()
                for o in y:
                    path_to_output = os.path.join("test_out",model_name,f"test-{observation_counters[model_name]}.wav")
                    torchaudio.save(path_to_output,o,SAMPLE_RATE, backend="soundfile")
                    observation_counters[model_name] += 1



infer(train_clean_dataloader,train_noise_dataloader)
infer(test_clean_dataloader,test_noise_dataloader)