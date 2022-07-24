import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Module, MSELoss
from torch.utils.tensorboard import SummaryWriter
from pretrainer.vae import VAE


class Pretrainer():

    def __init__(self, model: Module, device, vae:VAE, tensorlog_dir='./logdir'):
        self.model = model
        self.device = device
        self.vae = vae
        self.vae.to(device)
        self.model.to(device)
        self.sw = SummaryWriter(tensorlog_dir)


    def train(self, epoch, train_data, optimizer, criteria):
        train_batch_loss = 0.0
        train_epoch_loss = 0.0
        for i, (data, target) in enumerate(train_data):
            data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            r = self.model(data)
            loss = criteria(r, target)
            loss.backward()
            optimizer.step()
            train_batch_loss += loss.item()
            train_epoch_loss += loss.item()
            if i % 500 == 0:
                print(f"[epoch:{epoch}, {i:4d}] loss: {train_batch_loss / 500: .4f}")
                train_batch_loss = 0.0
        train_epoch_loss /= len(train_data)
        self.sw.add_scalar('Loss/train',train_epoch_loss, epoch)

    def evaluate(self, epoch, test_data, optimizer, criteria):
        evaluate_batch_loss = 0.0
        evaluate_epoch_loss = 0.0
        for i, (data, target) in enumerate(test_data):
            data.to(self.device), target.to(self.device)
            latent = self.vae.encode(data)
            r = self.model(latent)
            loss = criteria(r, target)
            evaluate_batch_loss += loss.item()
            if i % 500 == 0:
                print(f"[epoch:{epoch}, {i:4d}] loss: {evaluate_batch_loss / 500: .4f}")
                evaluate_batch_loss = 0.0
        evaluate_epoch_loss /= len(test_data)
        self.sw.add_scalar('Loss/eval',evaluate_epoch_loss, epoch)

    def start_training(self, epochs, train_data, test_data, lr=1e-3):
        optimizer = Adam(self.model.parameters(), lr=lr)
        criteria = MSELoss()
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            self.train(epoch, train_data, optimizer, criteria)
            self.model.eval()
            self.evaluate(epoch, test_data, optimizer, criteria)
