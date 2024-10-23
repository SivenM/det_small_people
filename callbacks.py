import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
import utils


class CSVLogger:
    def __init__(self, dir_path):
        utils.mkdir(dir_path)
        self.filepath = os.path.join(dir_path, 'logs.csv')
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'val_loss', 'acc', 'val_acc'])

    def log(self, epoch:int, loss:float, val_loss:float, acc:float, val_acc:float):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, val_loss, acc, val_acc])


class ModelCheckpoint:
    def __init__(self, dir_path, monitor='val_loss', mode='min', save_best_only=True):
        utils.mkdir(dir_path)
        self.filepath = os.path.join(dir_path, 'best_loss_{:.2f}.pth')
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None
        if mode == 'min':
            self.best = float('inf')
        elif mode == 'max':
            self.best = float('-inf')
        else:
            raise ValueError("Mode should be either 'min' or 'max'")

    def save(self, model, epoch:int, current:float):
        if self.save_best_only:
            if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
                self.best = current
                torch.save(model.state_dict(), self.filepath.format(current))
                print(f"Model improved at epoch {epoch+1}. Saving model to {self.filepath.format(current)}")
        else:
            torch.save(model.state_dict(), self.filepath.format(current))
            print(f"Saving model at epoch {epoch+1} to {self.filepath.format(current)}")


class TensorBoard:
    def __init__(self, log_dir:str, metrics_names:list=['Loss', 'Acc']) -> None:
        self.writer=SummaryWriter(log_dir)
        self.metrics_names = self.get_metric_names(metrics_names)

    def get_metric_names(self, in_names:list):
        out_names = []
        for name in in_names:
            out_names.append(name + '/train')
            out_names.append(name + '/val')
        return out_names

    def add(self, data:list, epoch:int) -> None:
        for i, info in enumerate(data):
            self.writer.add_scalar(self.metrics_names[i], info, epoch)

    def close(self):
        self.writer.close()