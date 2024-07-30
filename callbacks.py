import csv
import torch
from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'val_loss', 'acc', 'val_acc'])

    def log(self, epoch:int, loss:float, val_loss:float, acc:float, val_acc:float):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, val_loss, acc, val_acc])


class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
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
                torch.save(model.state_dict(), self.filepath)
                print(f"Model improved at epoch {epoch+1}. Saving model to {self.filepath}")
        else:
            torch.save(model.state(), self.filepath)
            print(f"Saving model at epoch {epoch+1} to {self.filepath}")


class TensorBoard:
    def __init__(self, log_dir:str, metrics_names:list) -> None:
        self.writer=SummaryWriter(log_dir)
        self.metrics_names = metrics_names

    def add(self, data:list, epoch:int) -> None:
        for i, info in enumerate(data):
            self.writer.add_scalar(self.metrics_names[i], info, epoch)

    def close(self):
        self.writer.close()