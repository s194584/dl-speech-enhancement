# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:16:30 2023

@author: jonas
"""

import os
import torch
import argparse
import numpy as np
import soundfile as sf
from utils.audiodec import AudioDec, assign_model
from torchaudio.functional import add_noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="libritts_v1")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-n", "--noise", type=str, required=False)
    parser.add_argument('--cuda', type=int, default=0 )
    parser.add_argument('--num_threads', type=int, default=4)
    args = parser.parse_args()

    # device assignment
    if args.cuda < 0:
        tx_device = f'cpu'
        rx_device = f'cpu'
    else:
        tx_device = f'cuda:{args.cuda}'
        rx_device = f'cuda:{args.cuda}'
    torch.set_num_threads(args.num_threads)

    # model assignment
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(args.model)

    # AudioDec initinalize
    print("AudioDec initinalizing!")
    audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

    with torch.no_grad():
        if os.path.exists(args.input):
            x_data, fs = sf.read(args.input, always_2d=True)
        else:
            raise ValueError(f'Input file {args.input} does not exist!')
        assert fs == sample_rate, f"data ({fs}Hz) is not matched to model ({sample_rate}Hz)!"
        x_data = torch.tensor(x_data, dtype=torch.float)  # (T, C)
        
        # Get noise
        if os.path.exists(args.noise):
            x_noise, fs_noise = sf.read(args.noise, always_2d=True)
        else:
            raise ValueError(f'Input file {args.noise} does not exist!')
        assert fs == fs_noise, f"data ({fs}Hz) is not matched to noise ({fs_noise}Hz)!"
        # x_noise = np.expand_dims(x_noise.transpose(1, 0), axis=1)
        x_noise = torch.tensor(x_noise, dtype=torch.float)  # (T, C)
        
        # Implement "trim_noise" function (for now we simply just trim)
        x_noise = x_noise[:x_data.shape[0], :]
        print(f"data shape (T, C): {x_data.shape}, noise shape (T, C): {x_noise.shape}")
        # Add noise to input
        snr = torch.tensor([5])
        x = add_noise(x_data.T, x_noise.T, snr).T
        print(f"data w/ noise shape (T, C): {x.shape}")
        x = x.numpy(force=True)
        x = np.expand_dims(x.transpose(1, 0), axis=1) # (T, C) -> (C, 1, T)
        x = torch.tensor(x).to(tx_device)
        
        sf.write(
            "noise_" + args.output,
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
            args.output,
            y,
            fs,
            "PCM_16",
        )
        print(f"Output {args.output}!")



if __name__ == "__main__":
    main()
