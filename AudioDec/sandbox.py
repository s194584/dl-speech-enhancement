import os

import torch
import soundfile as sf
import numpy as np
import torchaudio

from models.autoencoder_without_PQC.AudioDec import Generator as generator_audiodec
import math
import yaml

from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
import mir_eval
import soundfile as sf
import librosa
import os
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch.nn.modules.loss import _Loss
from torchaudio import transforms
from torch import nn, functional
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, \
    ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality




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
    clean_audio, clean_sr = torchaudio.load(clean_file_path)
    noise_audio, noise_sr = torchaudio.load(noise_file_path)

    if clean_audio.shape[1] > noise_audio.shape[1]:
        clean_audio = torch.narrow(clean_audio, 1, 0, noise_audio.shape[1])
    else:
        noise_audio = torch.narrow(noise_audio, 1, 0, clean_audio.shape[1])

    clean_audio = torchaudio.functional.resample(clean_audio,clean_sr,SAMPLE_RATE)
    noise_audio = torchaudio.functional.resample(noise_audio,noise_sr,SAMPLE_RATE)

    audio = add_noise(clean_audio, noise_audio, 10)

    torchaudio.save(mixed_output_path, audio, SAMPLE_RATE)

    audio_t = np.expand_dims(audio, axis=0)  # (T, C) -> (C, 1, T)
    # audio_t = np.expand_dims(audio_t, axis=0)  # (T, C) -> (C, 1, T)
    with torch.no_grad():
        audio_t = torch.tensor(audio_t).to(device)
        y = model(audio_t,clean_sr)[0]
        torchaudio.save(pred_output_path, y, SAMPLE_RATE)
    return clean_audio, audio, y

SAMPLE_RATE = 48000
# Load model
model = "vctk_denoise"
path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")

config = load_config(path_to_config)
generator_model = generator_audiodec(model_sample_rate=48000,**config["generator_params"])

tag = "Mel_Shape"
checkpoint = 59552
model_checkpoint = os.path.join("exp", "denoise", f"Mel_Shape_checkpoint-59552.pkl")


state_dict = torch.load(model_checkpoint, map_location="cpu")
generator_model.load_state_dict(state_dict)
encoder = generator_model.encoder
encoder.to(device=device)
decoder = generator_model.decoder
decoder.to(device=device)

clean, mixed, pred = reconstruct_test(generator_model, "cpu", checkpoint, tag)

def plot_specgram(waveforms, sample_rate, title, xlim=None):
    figure, axes = plt.subplots(len(waveforms), 1)
    for c, waveform in enumerate(waveforms):
        waveform = waveform.numpy()

        axes[c].specgram(waveform[0], Fs=sample_rate)
        axes[c].set_title(title[c])
    plt.show(block=False)


mel_spectrogram = transforms.MelSpectrogram(48000)
mae = nn.L1Loss()
def Mel_L1(pred, target):
    pred_mel = mel_spectrogram(pred)
    target_mel = mel_spectrogram(target)

    return mae(pred_mel,target_mel)


def print_measures(waveform1, waveform2):
    measures = {
        'MAE': nn.L1Loss(),
        'MSE': nn.MSELoss(),
        'SNR': SignalNoiseRatio(),
        'SDR': SignalDistortionRatio(),
        'SI-SDR(True)': ScaleInvariantSignalDistortionRatio(zero_mean=True),
        'SI-SDR(False)': ScaleInvariantSignalDistortionRatio(zero_mean=False),
        'PESQ': PerceptualEvaluationSpeechQuality(fs=16000, mode='wb'),
        'STOI': ShortTimeObjectiveIntelligibility(48000),
        'Mel-L1': Mel_L1,
    }
    for measure_name, measure in measures.items():
        if measure_name == 'PESQ':
            resampler = transforms.Resample(48000, 16000)
            print(f'{measure_name}: {measure(resampler(waveform1), resampler(waveform2))}')
        else:
            print(f'{measure_name}: {measure(waveform1, waveform2)}')


waveform_clean = torchaudio.functional.resample(clean,SAMPLE_RATE,48000)
waveform_mixed = torchaudio.functional.resample(mixed,SAMPLE_RATE,48000)
waveform_pred = torchaudio.functional.resample(pred,SAMPLE_RATE,48000)
waveform_clean = torch.narrow(waveform_clean, 1, 0, waveform_pred.shape[1])

# spectrogram = transform(waveform)
plot_specgram([waveform_clean, waveform_mixed, waveform_pred], 48000, ["Clean", "Mixed", "Prediction"])

def plot_waveform(waveforms, sample_rate, title="Waveform", xlim=None, ylim=None):
    figure, axes = plt.subplots(len(waveforms), 1)

    for c, waveform in enumerate(waveforms):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate
        axes[c].plot(time_axis, waveform[0], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
          axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
          axes[c].set_xlim(xlim)
        if ylim:
          axes[c].set_ylim(ylim)
        figure.suptitle(title)
    plt.show(block=False)

plot_waveform([waveform_clean, waveform_mixed, waveform_pred],48000,"Waveforms")







