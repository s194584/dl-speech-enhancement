import AudioDec
from utils.audiodec import AudioDec, assign_model
from dataloading.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch
import math
import soundfile as sf
import os
import numpy as np
from plot_audio_functions import plot_specgram
from trainer.denoise import Trainer as DenoiseTrainer

ENVIRONMENT = 'LAPTOP'
# ENVIRONMENT = 'HPC'

if ENVIRONMENT == 'LAPTOP':
    CLEAN_PATH = "corpus/train/clean"
    NOISE_PATH = "corpus/train/noise"
elif ENVIRONMENT == 'HPC':
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"


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


def define_splits(dataset, generator, train_percentage=0.7, val_percentage=0.15, test_percentage=0.15):
    split_datasets = random_split(dataset, [train_percentage, val_percentage, test_percentage], generator=generator)
    return {'train': split_datasets[0], 'val': split_datasets[1], 'test': split_datasets[2]}


# Seeds for reproducibility #########
generator_seed = 81
generator = torch.Generator().manual_seed(generator_seed)

# Loading model #####################
# TODO - Load model
model = 'vctk_denoise'

# device assignment
if -1 < 0:
    tx_device = f'cpu'
    rx_device = f'cpu'
else:
    tx_device = f'cuda:{args.cuda}'
    rx_device = f'cuda:{args.cuda}'
torch.set_num_threads(1)
model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)

print("AudioDec initinalizing!")
audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

# Loading data ######################
sample_rate = 48000
assert sample_rate == model_sample_rate, "The model's sample rate is not 48000Hz."
batch_length = 150000
snr = [20, 10, 3]
clean_dataset = AudioDataset(CLEAN_PATH, batch_length, sample_rate)
noise_dataset = AudioDataset(NOISE_PATH, batch_length, sample_rate)

clean_splits = define_splits(clean_dataset, generator)
noise_splits = define_splits(noise_dataset, generator)

train_clean_dataloader = DataLoader(clean_splits['train'], batch_size=1, shuffle=True, generator=generator)
train_noise_dataloader = DataLoader(noise_splits['train'], batch_size=1, shuffle=True, generator=generator)

# Loading Trainer ###################

# Training loop #####################
for i_batch, clean_sample_batch in enumerate(iter(train_clean_dataloader)):
    if i_batch == 2:
        break
    for j_batch, noise_sample_batch in enumerate(iter(train_noise_dataloader)):
        if j_batch == 2:
            break
        
        # Make pseudo batch
        for clean_sample in clean_sample_batch:
            for noise_sample in noise_sample_batch:
                x_noisy = add_noise(clean_sample, noise_sample, snr=10)
                x_target = clean_sample

        # Perform training on pseudo batch
        


                with torch.no_grad():
                    x = np.expand_dims(x_noisy.transpose(1, 0), axis=1)  # (T, C) -> (C, 1, T)
                    x = torch.tensor(x, dtype=torch.float).to(tx_device)
                    print("Encode/Decode...")
                    z = audiodec.tx_encoder.encode(x)
                    idx = audiodec.tx_encoder.quantize(z)
                    zq = audiodec.rx_encoder.lookup(idx)
                    y = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
                    y = y.squeeze(1).transpose(1, 0).cpu().numpy()  # T x C

                sf.write(
                    f"test-output/clean{i_batch}-noise{j_batch}_pred.wav",
                    y,
                    sample_rate,
                    "PCM_16",
                )
                sf.write(
                    f"test-output/clean{i_batch}-noise{j_batch}.wav",
                    x_noisy,
                    sample_rate,
                    "PCM_16",
                )
