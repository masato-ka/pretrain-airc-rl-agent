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

from pretrainer.vae import VAE


class DonkeyDataset(Dataset):

    def __init__(self, path_to_datasets, transforms=None, vae:VAE=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.labels = []
        self.images = []
        self.transforms = transforms
        self.vae = vae
        self.vae.eval()
        self.device=device
        labels_path_pattern = os.path.abspath(os.path.join(path_to_datasets, 'record_*.json'))
        data_path_pattern = os.path.abspath(os.path.join(path_to_datasets, '*.jpg'))

        for path in glob(labels_path_pattern):
            self.labels.append(path)
        for path in glob(data_path_pattern):
            self.images.append(path)
        self.label = sorted(self.labels, key=lambda x:int(re.sub(r'\D', "", os.path.basename(x))))
        self.image = sorted(self.images, key=lambda x:int(re.sub(r'\D', "", os.path.basename(x))))

    def __len__(self):
        return len(self.images)

    def _get_telemetry(self, file_path):
        with open(file_path, 'r') as f:
            telem = json.load(f)
            throttle = telem['user/throttle']
            angle = telem['user/angle']
        return (angle, throttle)

    def _get_history(self, idx):
        offset = 20 if idx > 20 else 20-idx
        history = [self._get_telemetry(p) for p in self.labels[idx-offset:idx]]
        history = np.asarray(history).astype(np.float32)
        history = history.reshape((len(history)*2))
        padding = (40 - len(history)) if len(history) < 40 else 0
        if len(history) < 40:
            padding = 40 - len(history)
            padding_data = np.zeros(padding).astype(np.float32)
            history = np.hstack((padding_data, history))
        return history


    def _preprocess_to_obs(self, image, idx):

        latent, _, _ = self.vae.encode(torch.unsqueeze(image, dim=0).to(self.device))
        latent = torch.squeeze(latent)
        #TODO make action history from json file.
        history = self._get_history(idx)
        history = torch.Tensor(history)
        latent = torch.cat([latent.detach(), history], dim=0)
        latent.detach().cpu().numpy()
        return latent


    def __getitem__(self, idx):
        label_file_path = self.labels[idx]
        image_file_path = self.images[idx]

        image = Image.open(image_file_path)
        if self.transforms:
            image = self.transforms(image)
        else:
            image = np.array(image).astype(np.float32).transpose(2,1,0)
            torchvision.transforms.ToTensor(image)
        obs = self._preprocess_to_obs(image.to(self.device), idx)
        telemetry = self._get_telemetry(label_file_path)
        #TODO 結果をcacheしたい。python3.9ならcacheデコレータつかえる？
        return obs, torch.Tensor(telemetry)



if __name__ == '__main__':
    d = DonkeyDataset("", transforms=transforms.Compose([
        torchvision.transforms.Resize((120, 160)),
        torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
        transforms.ToTensor(),
    ]))

    r=0.3

    train_size = int(len(d)*0.7)
    test_size = int(len(d)*0.3)
    adjust = len(d) - (train_size+test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(
        d, [train_size, test_size+adjust]
    )

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

    for i, (data, target) in enumerate(test_dataloader):
        print(data)
        print(target)
