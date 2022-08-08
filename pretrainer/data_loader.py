import os
import re
from glob import glob
import json

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class JetBotDataset(Dataset):

    def __init__(self, path_to_datasets, transforms=None, history_size=20,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.labels = []
        self.latents = []
        self.transforms = transforms
        self.device=device
        self.history_size = history_size
        labels_path_pattern = os.path.abspath(os.path.join(path_to_datasets, '*.jpg'))
        data_path_pattern = os.path.abspath(os.path.join(path_to_datasets, '*.jpg.npy'))

        for path in glob(labels_path_pattern):
            self.labels.append(path)
        for path in glob(data_path_pattern):
            self.latents.append(path)

        self.labels = sorted(self.labels, key=lambda x:os.path.basename(x).split("_")[2].split('.')[0])
        self.latents = sorted(self.latents, key=lambda x:os.path.basename(x).split("_")[2].split('.')[0])

    def __len__(self):
        return len(self.latents)

    def _get_telemetry(self, file_path):
        file_name = os.path.basename(file_path)
        throttle = float(file_name.split('_')[0])
        angle = float(file_name.split('_')[1])
        return angle, throttle

    def _get_history(self, idx, history_size=20):
        offset = history_size if idx > history_size else history_size-idx
        history = [self._get_telemetry(p) for p in self.labels[idx-offset:idx]]
        history = np.asarray(history).astype(np.float32)
        history = history.reshape((len(history)*2))
        padding = ((history_size*2) - len(history)) if len(history) < (history_size*2) else 0
        if len(history) < (history_size*2):
            padding = (history_size*2) - len(history)
            padding_data = np.zeros(padding).astype(np.float32)
            history = np.hstack((padding_data, history))
        return history


    def _preprocess_to_obs(self, latent, idx, history_size=20):
        latent = torch.Tensor(latent)
        latent = torch.squeeze(latent)
        history = self._get_history(idx, history_size)
        history = torch.Tensor(history)
        latent = torch.cat([latent.detach(), history], dim=0)
        latent.detach().cpu().numpy()
        return latent


    def __getitem__(self, idx):
        label_file_path = self.labels[idx]
        latent_file_path = self.latents[idx]

        latent = np.load(latent_file_path)
        obs = self._preprocess_to_obs(latent, idx, self.history_size)
        telemetry = self._get_telemetry(label_file_path)
        #TODO 結果をcacheしたい。python3.9ならcacheデコレータつかえる？
        return obs, torch.Tensor(telemetry)


class DonkeyDataset(Dataset):

    def __init__(self, path_to_datasets, transforms=None, history_size=20,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.labels = []
        self.latents = []
        self.transforms = transforms
        self.device=device
        self.agent_min_throttle = 0.6
        self.agent_max_throttle = 0.9
        self.history_size = history_size
        labels_path_pattern = os.path.abspath(os.path.join(path_to_datasets, 'record_*.json'))
        data_path_pattern = os.path.abspath(os.path.join(path_to_datasets, '*.jpg.npy'))

        for path in glob(labels_path_pattern):
            self.labels.append(path)
        for path in glob(data_path_pattern):
            self.latents.append(path)

        self.labels = sorted(self.labels, key=lambda x:int(re.sub(r'\D', "", os.path.basename(x))))
        self.latents = sorted(self.latents, key=lambda x:int(re.sub(r'\D', "", os.path.basename(x))))

    def __len__(self):
        return len(self.latents)

    def _get_telemetry(self, file_path):
        with open(file_path, 'r') as f:
            telem = json.load(f)
            throttle = telem['user/throttle']
            angle = telem['user/angle']
        return angle, throttle

    def _get_history(self, idx, history_size=20):
        offset = history_size if idx > history_size else history_size-idx
        history = [self._get_telemetry(p) for p in self.labels[idx-offset:idx]]
        history = np.asarray(history).astype(np.float32)
        history = history.reshape((len(history)*2))
        padding = ((history_size*2) - len(history)) if len(history) < 40 else 0
        if len(history) < (history_size*2):
            padding = (history_size*2) - len(history)
            padding_data = np.zeros(padding).astype(np.float32)
            history = np.hstack((padding_data, history))
        return history


    def _preprocess_to_obs(self, latent, idx, history_size=20):
        latent = torch.Tensor(latent)
        latent = torch.squeeze(latent)
        history = self._get_history(idx, history_size)
        history = torch.Tensor(history)
        latent = torch.cat([latent.detach(), history], dim=0)
        latent.detach().cpu().numpy()
        return latent


    def __getitem__(self, idx):
        label_file_path = self.labels[idx]
        latent_file_path = self.latents[idx]

        latent = np.load(latent_file_path)
        obs = self._preprocess_to_obs(latent, idx, self.history_size)
        telemetry = self._get_telemetry(label_file_path)
        #TODO 結果をcacheしたい。python3.9ならcacheデコレータつかえる？
        return obs, torch.Tensor(telemetry)



if __name__ == '__main__':
    d = JetBotDataset("/Users/kawamuramasato/Desktop/dataset", transforms=transforms.Compose([
        torchvision.transforms.Resize((120, 160)),
        torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
        transforms.ToTensor(),
    ]), history_size=0)

    r=0.3

    train_size = int(len(d)*0.7)
    test_size = int(len(d)*0.3)
    adjust = len(d) - (train_size+test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(
        d, [train_size, test_size+adjust]
    )

    print(train_dataset[0])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    # for i, (data, target) in enumerate(test_dataloader):
    #     print(data)
    #     print(target)
