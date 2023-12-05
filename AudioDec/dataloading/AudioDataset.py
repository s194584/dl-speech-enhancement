import os
import random

import torchaudio
from torch.utils.data import Dataset
import soundfile as sf
import glob


class AudioDataset(Dataset):
    def __init__(self, audio_dir, audio_root, batch_length, sample_rate):
        self.audio_dir = audio_dir
        self.batch_length = batch_length
        self.sample_rate = sample_rate
        self.audio_file_names = []
        for i in range(1, 3):
            layers = "/*" * i
            files = glob.glob(audio_dir + layers + ".wav")
            self.audio_file_names.extend(
                [
                    filename.replace("\\", "/").split(audio_root + "/")[-1]
                    for filename in files
                ]
            )

    def __len__(self):
        return len(self.audio_file_names)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_file_names[idx])
        audio, original_sample_rate = torchaudio.load(audio_path, backend="soundfile")
        audio = (
            torchaudio.functional.resample(
                audio, original_sample_rate, self.sample_rate
            )
            .transpose(1, 0)
            .detach()
            .numpy()
        )
        return audio
