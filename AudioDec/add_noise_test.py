# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:16:10 2023

@author: jonas
"""

import os
import torch
import argparse
import numpy as np
import soundfile as sf
from utils.audiodec import AudioDec, assign_model
from torchaudio.functional import add_noise
from plot_audio_functions import plot_waveform, plot_specgram
# from plot_audio_functions import add_noise

model = "vctk_v1"
inp = "input.wav"
out = "out.wav"
noise = "corpus/train/noisy/__CKGT4fj_Y.wav"
cuda = 0
num_threads = 4
# device assignment
if cuda < 0:
    tx_device = f'cpu'
    rx_device = f'cpu'
else:
    tx_device = f'cuda:{cuda}'
    rx_device = f'cuda:{cuda}'
torch.set_num_threads(num_threads)

# model assignment
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)

# AudioDec initinalize
print("AudioDec initinalizing!")
audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

with torch.no_grad():
    if os.path.exists(inp):
        x_data, fs = sf.read(inp, always_2d=True)
    else:
        raise ValueError(f'Input file {inp} does not exist!')
    assert fs == sample_rate, f"data ({fs}Hz) is not matched to model ({sample_rate}Hz)!"
    x_data = torch.tensor(x_data, dtype=torch.float)  # (T, C)
    
    
    plot_waveform(x_data.T, fs, title="Original", xlim=(0, 2))
    plot_specgram(x_data.T, fs, title="Original", xlim=(0, 2))
    
    # Get noise
    if os.path.exists(noise):
        x_noise, fs_noise = sf.read(noise, always_2d=True)
    else:
        raise ValueError(f'Input file {noise} does not exist!')
    assert fs == fs_noise, f"data ({fs}Hz) is not matched to noise ({fs_noise}Hz)!"
    # x_noise = np.expand_dims(x_noise.transpose(1, 0), axis=1)
    x_noise = torch.tensor(x_noise, dtype=torch.float)  # (T, C)
    
    plot_waveform(x_noise.T, fs, title="Noise", xlim=(0, 2))
    plot_specgram(x_noise.T, fs, title="Noise", xlim=(0, 2))
    
    
    # Implement "trim_noise" function (for now we simply just trim)
    x_noise = x_noise[:x_data.shape[0], :]
    print(f"data shape (T, C): {x_data.shape}, noise shape (T, C): {x_noise.shape}")
    # Add noise to input
    snr = torch.tensor([20])
    # snr = 20
    x = add_noise(x_data.T, x_noise.T, snr).T
    
    
    data_with_noise = x
    print(f"data w/ noise shape (T, C): {x.shape}")
    x = x.numpy(force=True)
    x = np.expand_dims(x.transpose(1, 0), axis=1) # (T, C) -> (C, 1, T)
    x = torch.tensor(x).to(tx_device)
    
    plot_waveform(data_with_noise.T, fs, title="data_with_noise", xlim=(0, 2))
    plot_specgram(data_with_noise.T, fs, title="data_with_noise", xlim=(0, 2))
    
    sf.write(
        "noise_" + out,
        x.squeeze(1).transpose(1, 0).cpu().numpy(),
        fs,
        "PCM_16",
    )
    
    print("Encode/Decode...")
    z = audiodec.tx_encoder.encode(x)
    idx = audiodec.tx_encoder.quantize(z)
    zq = audiodec.rx_encoder.lookup(idx)
    y = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
    y = y.squeeze(1).transpose(1, 0).cpu().numpy() # T x C
    sf.write(
        out,
        y,
        fs,
        "PCM_16",
    )
    print(f"Output {out}!")
