from torchvision import datasets, transforms
import torchaudio
import torch
import pandas as pd
import numpy as np
import glob
import os
import subprocess
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.utils.data import Dataset
from omegaconf import DictConfig
from ssm.utils import DatasetRegistry

@DatasetRegistry.register("smnist")
class SMNIST(Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root=cfg.path, train=cfg.train, download=True, transform=self.transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img.flatten(), label

@DatasetRegistry.register("audiomnist")
class AudioMNIST(Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.path = cfg.path
        self.destination = 'train' if cfg.train else 'val'
        if not os.path.exists(os.path.join(self.path, "AudioMNIST")):
            os.makedirs(os.path.join(self.path, "AudioMNIST"), exist_ok=True)
            self._download()

        self.wav_files = glob.glob(f"{self.path}/AudioMNIST/data/*/*.wav", recursive=True)
        np.random.seed(42)
        np.random.shuffle(self.wav_files)
        self.wav_files_train = self.wav_files[:int(len(self.wav_files) * 0.8)]
        self.wav_files_val = self.wav_files[int(len(self.wav_files) * 0.8):]

    def _download(self):
        subprocess.run(
            f"curl -L -o {self.path}/AudioMNIST.zip https://www.kaggle.com/api/v1/datasets/download/sripaadsrinivasan/audio-mnist && unzip {self.path}/AudioMNIST.zip -d {self.path}/AudioMNIST/",
            shell=True,
            check=True
        )

    def __len__(self):
        if self.destination == 'train':
            return len(self.wav_files_train)
        else:
            return len(self.wav_files_val)

    def __getitem__(self, idx):
        if self.destination == 'train':
            wav_path = self.wav_files_train[idx]
        else:
            wav_path = self.wav_files_val[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        label = int(os.path.basename(wav_path).split('_')[0])

        return waveform.squeeze(0)[:14073], label

@DatasetRegistry.register("weather")
class WeatherDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        """
        data: Pandas DataFrame containing weather data (features only, no timestamps)
        window_size: Number of past time steps to use for prediction
        """
        self.window_size = cfg.window_size
        self.train = cfg.train
        self.mask_idx = self.window_size // 2
        self.data = pd.read_csv(f"{cfg.path}/weather/WTH.csv").values[:, 1:].astype(np.float32)
        self.train_data = self.data[:int(len(self.data) * 0.8)]
        self.test_data = self.data[int(len(self.data) * 0.8):]
        self.num_features = self.data.shape[1]

    def __len__(self):
        if self.train:
            return len(self.train_data) - self.window_size
        else:
            return len(self.test_data) - self.window_size

    def __repr__(self):
        return f"WeatherDataset(window_size={self.window_size}, train={self.train})"

    def __getitem__(self, idx):
        if self.train:
            data = torch.tensor(self.train_data[idx:idx + self.window_size])
        else:
            data = torch.tensor(self.test_data[idx:idx + self.window_size])
        
        x, y  = torch.zeros(self.window_size, self.num_features), torch.zeros(self.window_size, self.num_features)
        breakpoint()
        x[:self.mask_idx] = data[:self.mask_idx]
        y[self.mask_idx:] = data[self.mask_idx:]
        return torch.tensor(x), torch.tensor(y) #(L, D), (L, D


if __name__ == "__main__":
    config = {
        "path": "./data",
        "train": True,
        "window_size": 48
    }
    cfg = SimpleNamespace(**config)
    # dataset_audio = AudioMNIST()
    # dataset_mnist = SMNIST()
    dataset_weather = WeatherDataset(cfg)