import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
import utils


class CSVLogger__:
    def __init__(self, dir_path):
        utils.mkdir(dir_path)
        self.filepath = os.path.join(dir_path, 'logs.csv')
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'val_loss', 'acc', 'precision', 'recall', 'val_acc', 'v_precision', 'v_recall'])

    def log(self, epoch:int, loss:float, val_loss:float, t_metrics:dict, v_metrics:dict):
        print("Вызван log, epoch =", epoch)
        row = [epoch, loss, val_loss, 
           t_metrics.get('acc'), t_metrics.get('precision'), t_metrics.get('recall'),
           v_metrics.get('acc'), v_metrics.get('precision'), v_metrics.get('recall')]
        print("Строка для записи:", row)
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            print("Записали строку → flush выполнен")

class CSVLogger:
    def __init__(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.filepath = os.path.join(dir_path, 'logs.csv')
        print("Логгер будет писать в →", os.path.abspath(self.filepath))  # ← сразу видно
        
        self.f = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['epoch', 'loss', 'val_loss', 'acc', 'precision', 'recall', 'val_acc', 'v_precision', 'v_recall'])
        self.f.flush()

    def log(self, epoch:int, loss:float, val_loss:float, t_metrics:dict, v_metrics:dict):
        row = [epoch, loss, val_loss, 
           t_metrics.get('acc'), t_metrics.get('precision'), t_metrics.get('recall'),
           v_metrics.get('acc'), v_metrics.get('precision'), v_metrics.get('recall')]
        self.writer.writerow(row)
        self.f.flush()
        #print("Записана эпоха", epoch, "→ размер файла теперь", os.path.getsize(self.filepath))

    def close(self):
        if not self.f.closed:
            self.f.flush()
            self.f.close()


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
    def __init__(self, log_dir:str, metrics_names:list=['Loss', 'Acc', 'IoU']) -> None:
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