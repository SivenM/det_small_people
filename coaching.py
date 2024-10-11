import os
from tqdm.auto import tqdm
from typing import Tuple
import torch
from torch import nn
from torcheval.metrics.functional import binary_accuracy, multiclass_accuracy
from callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import utils
from loguru import logger

BASE_DIR = "runs"


class Coach:

    def __init__(self, name:str, save_dir:str=None, metric:str='multi_acc', logger:bool=True, checkpoint:bool=True, tboard:bool=False, device='cuda', debug:bool=False) -> None:
        if save_dir:
            utils.mkdir(save_dir)
            utils.mkdir(f'{save_dir}/{name}')
            self.logs_dir = f'{save_dir}/{name}/logs'
            self.save_dir = f'{save_dir}/{name}/models/'
        else:
            utils.mkdir(f'{BASE_DIR}/{name}')
            self.logs_dir = f'{BASE_DIR}/{name}/logs'
            self.save_dir = f'{BASE_DIR}/{name}/models/'
            self.device = device
        
        if metric == 'multi_acc':
            self.metric = multiclass_accuracy
        elif metric == ' bin_acc':
            self.metric = binary_accuracy
        else:
            raise 'Wrong metric'
            
        if logger:
            self.logger = CSVLogger(self.logs_dir)
        else:
            self.logger = None

        if checkpoint:
            self.checkpoint = ModelCheckpoint(self.save_dir)
        else:
            self.checkpoint = None

        if tboard:
            self.tboard = TensorBoard(f'{save_dir}/{name}/tb')
        else:
            self.tboard = None
        self.history = {
            't_loss': [],
            'v_loss': [],
            't_acc': [],
            'v_acc': []
        }
        self.device = device
        self.debug = debug

    def __create_dirs(self, name):
        exp_dir = os.path.join(BASE_DIR, name)
        logs_dir = os.path.join(exp_dir, 'logs')
        models_dir = os.path.join(exp_dir, 'models')
        for path in [exp_dir, logs_dir, ]:
            utils.mkdir(os.path.join(BASE_DIR, name))

    def _to_cuda(self, targets:list) -> list:
        for target in targets:
            target['bbox'] = target['bbox'].to('cuda')
            target['labels'] = target['labels'].to('cuda')
        return targets
    
    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss = 0.
        t_acc = 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        for batch, (sample, label) in progress_bar:
            if torch.cat([v["bbox"] for v in label]).shape[0] == 0:
                continue
            self.optimizer.zero_grad()
            sample, label= sample.to(self.device), self._to_cuda(label)
            pred = self.model(sample)
            loss = self.loss_fn(pred, label)
            t_loss += loss.item()
            #if self.debug:
            #print(f'loss: {t_loss}')
            #t_acc += self.metric(pred, label['labels']).item()
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1),
                    #'train_acc': t_acc / (batch + 1)
                }
            )

        t_loss = t_loss / len(data)
        t_acc = t_acc / len(data)
        return t_loss, t_acc

    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc = 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                if torch.cat([v["bbox"] for v in label]).shape[0] == 0:
                    continue
                sample, label= sample.to(self.device), self._to_cuda(label)
                pred = self.model(sample)
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                #v_acc += self.metric(pred, label).item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                        #'val_acc': v_acc / (batch + 1)
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
            self.tboard.add([t_loss, v_loss], epoch)

    def plot_history(self):
        pass

    def fit(self, epoches, model, loss_fn, lr, train_data, val_data=None):
        self.clear_history()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in tqdm(range(epoches)):
            t_loss, t_acc = self.train_step(epoch, train_data)
            if val_data:
                v_loss, v_acc = self.val_step(epoch, val_data)
            else:
                v_loss, v_acc = 0., 0.
            self.update_callbacks(epoch, t_loss, v_loss, t_acc, v_acc)
            self.update_history(t_loss, v_loss, t_acc, v_acc)
            #if self.tboard:
            #    self.tboard.add_scalar('Loss/train', t_loss, epoch)
            #    self.tboard.add_scalar('Loss/train', v_loss, epoch)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last_acc{v_acc}.pth'))
        return self.history


class DetrLocCoach(Coach):
    def __init__(self, name: str, save_dir: str = None, metric: str = 'multi_acc', logger: bool = True, checkpoint: bool = True, tboard: bool = False, device='cuda', debug: bool = False) -> None:
        super().__init__(name, save_dir, metric, logger, checkpoint, tboard, device, debug)

    def _to_cuda(self, targets:list) -> list:
        out = []
        for target in targets:
            out.append(target.to('cuda'))
        return out

    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss, t_acc = 0., 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        for batch, (sample, label) in progress_bar:
            if torch.cat([v for v in label]).shape[0] == 0:
                continue
            self.optimizer.zero_grad()
            sample, label = sample.to(self.device), self._to_cuda(label)
            pred = self.model(sample)
            loss = self.loss_fn(pred, label)
            t_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss
                }
            )
        return t_loss, t_acc
    
    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc = 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                if torch.cat([v for v in label]).shape[0]  == 0:
                    continue
                sample, label= sample.to(self.device), self._to_cuda(label)
                pred = self.model(sample)
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss,
                    }
                )
        return v_loss, v_acc


class ObjCoach:

    def __init__(self) -> None:
        pass