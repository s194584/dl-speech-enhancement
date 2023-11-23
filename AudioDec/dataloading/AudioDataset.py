import os
import random

from torch.utils.data import Dataset
import soundfile as sf
import glob

class AudioDataset(Dataset):
    def __init__(self, audio_dir, audio_root, batch_length, sample_rate):
        self.audio_dir = audio_dir
        self.batch_length = batch_length
        self.sample_rate = sample_rate
        self.audio_file_names = []
        for i in range(1,3):
            layers = "/*"*i
            files = glob.glob(audio_dir + layers+".wav")
            self.audio_file_names.extend([filename.replace("\\","/").split(audio_root+"/")[-1] for filename in files])
        print("sfe")

    def __len__(self):
        return len(self.audio_file_names)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_file_names[idx])
        audio, _ = sf.read(audio_path, always_2d=True)
        # Make it match the batch_length
        # audio = audio[:self.batch_length]
        # TODO - Reconsider approach
        if len(audio) < self.batch_length:
            print(f'{audio_path}: {len(audio)}')
            return
        #     audio = np.pad(audio, [(self.batch_length - len(audio), 0), (0, 0)], constant_values=0)
        return audio

    def get_random(self):
        return self.__getitem__(random.randint(len(self.audio_file_names)))