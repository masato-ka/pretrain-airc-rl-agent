import argparse
import os
from glob import glob

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from stable_baselines3 import SAC
from torch.utils.data import DataLoader
from torchvision import transforms

from pretrainer.dummy_env import DummyEnv
from pretrainer.data_loader import DonkeyDataset, JetBotDataset
from pretrainer.pretrainer import Pretrainer
from pretrainer.utils import get_logger
from pretrainer.vae import VAE
import logging

logger = get_logger(__name__)


def prepare_dataset(dataset_folder, batch_size=64, test_rate=0.3, vae=None, device=None):
    logger.info(f'start preparing dataset.')
    transform = transforms.Compose([
        torchvision.transforms.Resize((120, 160), ),
        torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
        transforms.ToTensor(),
    ])
    logger.info('Image convert to latent spaces.')
    image_files = glob(os.path.abspath(os.path.join(dataset_folder, '*.jpg')))
    t_image_files = tqdm(image_files)
    for f in t_image_files:
        image = Image.open(f)
        image = transform(image)
        latent, _, _ = vae.encode(torch.unsqueeze(image.to(device), dim=0))
        latent = torch.squeeze(latent).detach().cpu().numpy()
        np.save(f + '.npy', latent)

    d = JetBotDataset(path_to_datasets=dataset_folder,
                      transforms=transform,
                      vae=vae)

    train_size = int(len(d) * (1 - test_rate))
    test_size = len(d) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        d, [train_size, test_size]
    )
    logger.info('train dataset size: {}'.format(len(train_dataset)))
    logger.info('test dataset size: {}'.format(len(test_dataset)))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )
    logger.info('complete prepare dataset.')
    return train_dataloader, test_dataloader


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--vae-pretrain', type=str, default='vae,pth')
    parser.add_argument('--use-sde', type=bool, default=False)
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_known_args()


def main():
    logger.info("start sac agent pretraining.")
    args, _ = _parse_args()

    device = None
    if args.use_cuda:
        logger.info('if your environment have cuda device, use cuda device.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        logger.info('use cpu device')
        device = torch.device('cpu')

    vae_model_path = args.vae_pretrain
    dataset_folder = args.train
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    save_path = os.path.join(args.model_dir, 'pretrain-sac.pth')

    logger.info('VAE pretrain model: {}'.format(vae_model_path))
    logger.info('Dataset load from: {}'.format(dataset_folder))
    logger.info('model artifact save path:'.format(save_path))
    logger.info('Epochs: {}'.format(epochs))
    logger.info('Batch size: {}'.format(batch_size))
    logger.info('Learning rate: {}'.format(lr))
    logger.info('SDE mode: {}'.format(args.use_sde))
    ## load_vae_model
    logger.info('loading vae pretrain model.')
    vae = VAE()
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.eval()
    logger.info('loaded vae model complete.')

    ## Prepare dataset.
    train_dataloader, test_dataloader = prepare_dataset(dataset_folder=dataset_folder, batch_size=batch_size, vae=vae)

    ## Prepare target model.
    logger.info('start create SAC policy model.')
    env = DummyEnv(z_dim=32)
    sac = SAC('MlpPolicy', env=env,
              policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32], use_sde=args.use_sde))
    model = sac.policy.to(device)
    logger.info('complete create SAC policy model.')


    ## Train
    logger.info('start training')
    trainer = Pretrainer(
        model, device
    )
    trainer.start_training(epochs=epochs, train_data=train_dataloader, test_data=test_dataloader, lr=lr)

    logger.info('save result to {}'.format(save_path))
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
