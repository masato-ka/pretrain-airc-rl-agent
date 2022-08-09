import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.nn import Module, MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pretrainer.vae import VAE


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0.0001):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.last_val_loss = None

    def __call__(self, train_loss, validation_loss):
        if self.last_val_loss is None:
            #Initialize
            self.last_val_loss = validation_loss
            return

        if (validation_loss - self.last_val_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        self.last_val_loss = validation_loss
        print(self.counter)

class Pretrainer():

    def __init__(self, model: Module, device, tensorlog_dir='./logdir'):
        self.model = model
        self.device = device
        self.train_data_list = None
        self.test_data_list = None
        self.sw = SummaryWriter(tensorlog_dir)
        self.early_stopping = EarlyStopping()

    def train(self, epoch, train_data, optimizer, criteria):
        train_epoch_loss = 0.0
        for i, (data, target) in enumerate(tqdm(train_data)):
            data = data.to(self.device);
            target = target.to(self.device)
            optimizer.zero_grad()
            r = self.model(data)
            loss = criteria(r, target)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        train_epoch_loss /= len(train_data)
        print(f"train:[epoch:{epoch}] loss: {train_epoch_loss: .4f}")
        self.sw.add_scalar('Loss/train', train_epoch_loss, epoch)
        return train_epoch_loss

    def evaluate(self, epoch, test_data, optimizer, criteria):
        evaluate_epoch_loss = 0.0
        for i, (data, target) in enumerate(tqdm(test_data)):
            data = data.to(self.device);
            target = target.to(self.device)
            r = self.model(data)
            loss = criteria(r, target)
            evaluate_epoch_loss += loss.item()
        evaluate_epoch_loss /= len(test_data)
        print(f"eval:[epoch:{epoch}] loss: {evaluate_epoch_loss: .4f}")
        self.sw.add_scalar('Loss/eval', evaluate_epoch_loss, epoch)
        return evaluate_epoch_loss

    def start_training(self, epochs, train_data, test_data, lr=1e-3):
        optimizer = Adam(self.model.parameters(), lr=lr)
        criteria = MSELoss()
        if self.train_data_list is None:
            self.train_data_list = list(train_data)
        if self.test_data_list is None:
            self.test_data_list = list(test_data)
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train(epoch, train_data, optimizer, criteria)
            self.model.eval()
            eval_loss = self.evaluate(epoch, test_data, optimizer, criteria)
            self.early_stopping(train_loss, eval_loss)
            if self.early_stopping.early_stop:
                print('Early stopping: epoch {}'.format(epoch))
                break
