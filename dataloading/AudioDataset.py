import os

import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
from os import listdir
from os.path import isfile, join


class AudioDataset(Dataset):
    def __init__(self, audio_dir, batch_length, sample_rate):
        self.audio_dir = audio_dir
        self.batch_length = batch_length
        self.sample_rate = sample_rate
        self.audio_file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    def __len__(self):
        return len(self.audio_file_names)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_file_names[idx])
        audio, _ = sf.read(audio_path, always_2d=True)
        # Make it match the batch_length
        audio = audio[:self.batch_length]
        # TODO - Reconsider approach
        if len(audio) < self.batch_length:
            audio = np.pad(audio, [(self.batch_length - len(audio), 0), (0, 0)], constant_values=0)
        return audio
