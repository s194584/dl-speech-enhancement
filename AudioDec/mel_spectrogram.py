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

print(sf.__version__)
print(sf.__libsndfile_version__)

path_clean = os.path.join('corpus', 'test', 'book_00002_chp_0005_reader_11980_3_seg_0.wav')
checkpoint=111661
path_mixed = f'job_out/MelL1_Adam-adjusted___{checkpoint}___mixed.wav'
path_pred = f'job_out/MelL1_Adam-adjusted___{checkpoint}___reconstructed.wav'

names = ['clean', 'mixed', 'reconstruction']
paths = [
    path_clean, path_mixed, path_pred
]


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


waveform_clean, sample_rate = torchaudio.load(path_clean)
waveform_mixed, sample_rate = torchaudio.load(path_mixed)
waveform_pred, sample_rate = torchaudio.load(path_pred)
waveform_clean = torch.narrow(waveform_clean, 1, 0, waveform_pred.shape[1])

# spectrogram = transform(waveform)
plot_specgram([waveform_clean, waveform_mixed, waveform_pred], 48000, ["Clean", "Mixed", "Prediction"])


print("\n- Clean vs. Clean-----")
print_measures(waveform_clean, waveform_clean)
print("\n- Mixed vs. Clean-----")
print_measures(waveform_mixed,waveform_clean)
print("\n- Prediction vs. Clean-----")
print_measures(waveform_pred, waveform_clean)
print("\n- Prediction vs. Mixed-----")
print_measures(waveform_pred, waveform_mixed)
