import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Module, MSELoss
from torch.utils.tensorboard import SummaryWriter
from pretrainer.vae import VAE


class Pretrainer():

    def __init__(self, model: Module, device, tensorlog_dir='./logdir'):
        self.model = model
        self.device = device
        self.sw = SummaryWriter(tensorlog_dir)


    def train(self, epoch, train_data, optimizer, criteria):
        train_batch_loss = 0.0
        train_epoch_loss = 0.0
        for i, (data, target) in enumerate(train_data):
            data = data.to(self.device); target = target.to(self.device)
            optimizer.zero_grad()
            r = self.model(data)
            loss = criteria(r, target)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        train_epoch_loss /= len(train_data)
        print(f"train:[epoch:{epoch}] loss: {train_batch_loss: .4f}")
        self.sw.add_scalar('Loss/train',train_epoch_loss, epoch)

    def evaluate(self, epoch, test_data, optimizer, criteria):
        evaluate_epoch_loss = 0.0
        for i, (data, target) in enumerate(test_data):
            data=data.to(self.device); target=target.to(self.device)
            r = self.model(data)
            loss = criteria(r, target)
            evaluate_epoch_loss += loss.item()
        evaluate_epoch_loss /= len(test_data)
        print(f"eval:[epoch:{epoch}] loss: {evaluate_epoch_loss: .4f}")
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
