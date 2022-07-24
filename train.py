import argparse
import os

import torch
import torchvision
from stable_baselines3 import SAC
from torch.utils.data import DataLoader
from torchvision import transforms

from pretrainer.DummyEnv import DummyEnv
from pretrainer.data_loader import DonkeyDataset
from pretrainer.pretrainer import Pretrainer
from pretrainer.vae import VAE


def prepare_dataset(dataset_folder, batch_size=64, test_rate=0.3, vae=None):
    d = DonkeyDataset(path_to_datasets=dataset_folder,
                      transforms=transforms.Compose([
                          torchvision.transforms.Resize((120, 160), ),
                          torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
                          transforms.ToTensor(),
                      ]),
                      vae=vae)

    train_size = int(len(d) * 1 - (test_rate))
    test_size = int(len(d) * test_rate)
    adjust = len(d) - (train_size + test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(
        d, [train_size, test_size + adjust]
    )

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

    return train_dataloader, test_dataloader


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--vae-pretrain', type=bool, default='vae,pth')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_known_args()


def main():
    args, _ = _parse_args()

    device = None
    if args.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    vae_model_path = args.vae_pretrain
    dataset_folder = args.train
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    save_path = args.model_dir

    ## load_vae_model
    vae = VAE()
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.eval()

    ## Prepare dataset.
    train_dataloader, test_dataloader = prepare_dataset(dataset_folder=dataset_folder, batch_size=batch_size, vae=vae)

    ## Prepare target model.
    env = DummyEnv(z_dim=32)
    sac = SAC('MlpPolicy', env=env, policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32]))
    model = sac.policy.to(device)

    ## Train
    trainer = Pretrainer(
        model, device, vae
    )
    trainer.start_training(epochs=epochs, train_data=train_dataloader, test_data=test_dataloader, lr=lr)

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
