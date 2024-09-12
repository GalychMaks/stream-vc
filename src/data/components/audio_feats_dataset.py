import glob
import os

import torch
from torch.utils.data import Dataset

from src.data.components.audio_utils import AudioUtils


class AudioFeatsDataset(Dataset):
    def __init__(self,
                 root_dir,
                 ext=".wav",
                 filelist=None,
                 ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.ext = ext

        if not filelist:
            filelist = glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
        else:
            try:
                with open(filelist, "r") as f:
                    filelist = f.readlines()
                    filelist = [os.path.join(root_dir, f.strip()) for f in filelist]
            except Exception as e:
                print(f"Error reading filelist: {e}")
                exit(-1)
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        fpath = self.filelist[idx]
        wav, sr = AudioUtils.load_audio(fpath, sample_rate=16000)
        wav = AudioUtils.to_mono(wav)

        pitch = torch.load(fpath.replace(self.ext, ".pitch.pt"), weights_only=True)
        energy = torch.load(fpath.replace(self.ext, ".energy.pt"), weights_only=True)
        hubert = torch.load(fpath.replace(self.ext, ".hubert.pt"), weights_only=True)

        feature_len = hubert.shape[-1]
        # excerpt 75 frames
        TARGET_LEN = 75  # The target sequence length

        if feature_len > TARGET_LEN:
            # Randomly sample a starting index to truncate the sequences
            start = torch.randint(0, feature_len - TARGET_LEN, (1,)).item()
            pitch = pitch[:, start:start + TARGET_LEN]
            energy = energy[:, start:start + TARGET_LEN]
            hubert = hubert[start:start + TARGET_LEN]
            wav = wav[:, start * 320:(start + TARGET_LEN) * 320]  # Adjust for the waveform's sample rate
        else:
            # Pad sequences to the target length (TARGET_LEN)
            pitch = torch.nn.functional.pad(pitch, (0, TARGET_LEN - pitch.shape[-1]), "constant", 0)
            energy = torch.nn.functional.pad(energy, (0, TARGET_LEN - energy.shape[-1]), "constant", 0)
            hubert = torch.nn.functional.pad(hubert, (0, TARGET_LEN - hubert.shape[-1]), "constant", 0)
            wav = torch.nn.functional.pad(wav, (0, TARGET_LEN * 320 - wav.shape[-1]), "constant", 0)

        return wav, pitch, energy, hubert
