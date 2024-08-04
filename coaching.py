import os
from tqdm.auto import tqdm
from typing import Tuple
import torch
from torch import nn
from torcheval.metrics.functional import binary_accuracy
from callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import utils

BASE_DIR = "runs"

class Coach:

    def __init__(self, name:str, logger:bool=True, checkpoint:bool=True, tboard:bool=False, device='cuda') -> None:
        utils.mkdir(f'{BASE_DIR}/{name}')
        self.logs_dir = f'{BASE_DIR}/{name}/logs'
        self.save_dir = f'{BASE_DIR}/{name}/models/'
        self.device = device
        
        if logger:
            self.logger = CSVLogger(self.logs_dir)
        else:
            self.logger = None

        if checkpoint:
            self.checkpoint = ModelCheckpoint(self.save_dir)
        else:
            self.checkpoint = None

        if tboard:
            self.tboard = TensorBoard(f'{BASE_DIR}/{name}/tb')
        else:
            self.tboard = None
        self.history = {
            't_loss': [],
            'v_loss': [],
            't_acc': [],
            'v_acc': []
        }

    def __create_dirs(self, name):
        exp_dir = os.path.join(BASE_DIR, name)
        logs_dir = os.path.join(exp_dir, 'logs')
        models_dir = os.path.join(exp_dir, 'models')
        for path in [exp_dir, logs_dir, ]:
            utils.mkdir(os.path.join(BASE_DIR, name))

        
    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        t_loss = 0.
        t_acc = 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        for batch, (sample, label) in progress_bar:
            self.optimizer.zero_grad()
            sample, label = sample.to(self.device), label.squeeze().to(self.device)
            pred = self.model(sample)
            loss = self.loss_fn(pred, label)
            t_loss += loss.item()
            t_acc += binary_accuracy(pred, label).item()
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1),
                    'train_acc': t_acc / (batch + 1)
                }
            )
        t_loss = t_loss / len(data)
        t_acc = t_acc / len(data)
        return t_loss, t_acc

    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        v_loss, v_acc = 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                sample, label = sample.to(self.device), label.squeeze().to(self.device)
                pred = self.model(sample)
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                v_acc += binary_accuracy(pred, label).item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                        'val_acc': v_acc / (batch + 1)
                    }
                )
        v_loss = v_loss / len(data)
        v_acc = v_acc / len(data)
        return v_loss, v_acc

    def clear_history(self):
        self.history['t_loss'] = []
        self.history['v_loss'] = []
        self.history['t_acc'] = []
        self.history['v_acc'] = []

    def update_history(self, t_loss, v_loss, t_acc, v_acc):
        self.history['t_loss'].append(t_loss)
        self.history['v_loss'].append(v_loss)
        self.history['t_acc'].append(t_acc)
        self.history['v_acc'].append(v_acc)

    def update_callbacks(self, epoch, t_loss, v_loss, t_acc, v_acc):
        if self.logger:
            self.logger.log(epoch, t_loss, v_loss, t_acc, v_acc)
        
        if self.checkpoint:
            self.checkpoint.save(self.model, epoch, v_loss)

        if self.tboard:
            self.tboard.add([t_loss, v_loss, t_acc, v_acc], epoch)

    def plot_history(self):
        pass

    def fit(self, epoches, model, loss_fn, optimizer, train_data, val_data=None):
        self.clear_history()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        for epoch in tqdm(range(epoches)):
            t_loss, t_acc = self.train_step(epoch, train_data)
            if val_data:
                v_loss, v_acc = self.val_step(epoch, val_data)
            else:
                v_loss, v_acc = 0., 0.
            self.update_callbacks(epoch, t_loss, v_loss, t_acc, v_acc)
            self.update_history(t_loss, v_loss, t_acc, v_acc)
        
        torch.save(model.state(), os.path.join(self.save_dir, f'last_acc{v_acc}.pth'))
        return self.history


class ObjCoach:

    def __init__(self) -> None:
        pass