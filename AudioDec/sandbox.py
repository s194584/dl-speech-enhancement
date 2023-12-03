import os

import torch
import soundfile as sf
import numpy as np
import torchaudio

from models.autoencoder.AudioDec import Generator as generator_audiodec
import math
import yaml

from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
import mir_eval


SAMPLE_RATE = 48000

def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def add_noise(speech, noise, snr=10):
    assert speech.shape == noise.shape, "Shapes are not equal!"

    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech


device = "cpu"



##########################################################################

def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)


def plot_mask(mask, title="Mask"):
    mask = mask.numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(mask, cmap="viridis", origin="lower", aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)


def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()


def generate_mixture(waveform_clean, waveform_noise, target_snr):
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise


def evaluate(estimate, reference):
    si_snr_score = si_snr(estimate, reference)
    (
        sdr,
        _,
        _,
        _,
    ) = mir_eval.separation.bss_eval_sources(reference.numpy(), estimate.numpy(), False)
    pesq_mix = pesq(SAMPLE_RATE, estimate[0].numpy(), reference[0].numpy(), "wb")
    stoi_mix = stoi(reference[0].numpy(), estimate[0].numpy(), SAMPLE_RATE, extended=False)
    print(f"SDR score: {sdr[0]}")
    print(f"Si-SNR score: {si_snr_score}")
    print(f"PESQ score: {pesq_mix}")
    print(f"STOI score: {stoi_mix}")

##########################################################################
def reconstruct_test(model, device, checkpoint,tag):
    clean_file_path = "corpus/test/book_00002_chp_0005_reader_11980_3_seg_0.wav"
    noise_file_path = "corpus/train/noise_single/__HpItICRe0.wav"
    pred_output_path = f"job_out/{tag}___{checkpoint}___reconstructed.wav"
    mixed_output_path = (
        f"job_out/{tag}___{checkpoint}___mixed.wav"
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

    encoder = model.encoder
    decoder = model.decoder

    audio_t = np.expand_dims(audio, axis=0)  # (T, C) -> (C, 1, T)
    audio_t = np.expand_dims(audio_t, axis=0)  # (T, C) -> (C, 1, T)
    with torch.no_grad():
        audio_t = torch.tensor(audio_t, dtype=torch.float).to(device)
        encoded_audio = encoder(audio_t)
        y = decoder(encoded_audio)[0].squeeze(1).transpose(1, 0).cpu().numpy()
        sf.write(
            pred_output_path,
            y,
            48000,
            "PCM_16",
        )



# Load model
model = "vctk_denoise"
path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")

config = load_config(path_to_config)
generator_model = generator_audiodec(**config["generator_params"])

tag = "MelL1_Adam-adjusted"
checkpoint = 111661
model_checkpoint = os.path.join("exp", "denoise", f"MelL1_Adam-adjusted_checkpoint-{checkpoint}.pkl")


state_dict = torch.load(model_checkpoint, map_location="cpu")
generator_model.load_state_dict(state_dict)
encoder = generator_model.encoder
encoder.to(device=device)
decoder = generator_model.decoder
decoder.to(device=device)

reconstruct_test(generator_model, "cpu", checkpoint, tag)
